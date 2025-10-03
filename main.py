import pymysql
import json
import re
import hashlib
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
import logging
import os
from pathlib import Path
from dotenv import load_dotenv
import time
import csv
import pandas as pd

load_dotenv()

# Ollama Integration
import ollama

# Vector Database
import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class ColumnInfo:
    """Enhanced column metadata with semantic understanding"""
    table_name: str
    column_name: str
    data_type: str
    is_nullable: bool = True
    is_primary: bool = False
    is_foreign: bool = False
    foreign_ref: Optional[str] = None
    column_comment: Optional[str] = None
    sample_values: List[str] = field(default_factory=list)
    semantic_type: Optional[str] = None

@dataclass
class TableInfo:
    """Table metadata with statistics"""
    table_name: str
    row_count: int = 0
    table_comment: Optional[str] = None
    columns: List[ColumnInfo] = field(default_factory=list)
    relationships: List[Dict] = field(default_factory=list)
    indexes: List[str] = field(default_factory=list)

class EnhancedDatabaseAnalyzer:
    """Comprehensive database analyzer with deep schema understanding"""
    
    def __init__(self, connection_params: dict, vector_db_path: str = "./vector_db"):
        self.conn_params = connection_params
        self.vector_db_path = Path(vector_db_path)
        self.vector_db_path.mkdir(parents=True, exist_ok=True)
        
        # Schema caches
        self._schema_cache: Dict[str, TableInfo] = {}
        self._last_refresh = None
        self.CACHE_DURATION = timedelta(hours=6)
        
        # Schema understanding status file
        self.schema_status_file = self.vector_db_path / "schema_status.json"
        
        # Initialize ChromaDB with persistence
        try:
            self.chroma_client = chromadb.PersistentClient(
                path=str(self.vector_db_path),
                settings=Settings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
            )
            
            # Create collections for different types of knowledge
            self.embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name="all-MiniLM-L6-v2"
            )
            
            self.schema_collection = self._get_or_create_collection("schema_knowledge", "Database schema and structure knowledge")
            self.query_collection = self._get_or_create_collection("query_history", "Historical queries and results")
            self.business_collection = self._get_or_create_collection("business_logic", "Business rules and domain knowledge")
            self.llm_schema_collection = self._get_or_create_collection("llm_schema_understanding", "LLM processed and understood schema")
            
            logger.info("✅ Vector DB initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB: {e}")
            raise

    def _get_or_create_collection(self, name: str, description: str):
        """Safely get or create a collection"""
        try:
            return self.chroma_client.get_or_create_collection(
                name=name,
                embedding_function=self.embedding_fn,
                metadata={"description": description}
            )
        except Exception as e:
            logger.error(f"Failed to create collection {name}: {e}")
            raise

    def get_connection(self):
        """Get database connection with error handling and retry"""
        max_retries = 3
        retry_delay = 1
        
        for attempt in range(max_retries):
            try:
                conn = pymysql.connect(
                    **self.conn_params,
                    connect_timeout=10,
                    read_timeout=30,
                    write_timeout=30
                )
                return conn
            except Exception as e:
                logger.warning(f"Connection attempt {attempt + 1} failed: {e}")
                if attempt < max_retries - 1:
                    time.sleep(retry_delay * (attempt + 1))
                else:
                    logger.error(f"Database connection failed after {max_retries} attempts")
                    raise

    def is_schema_analyzed(self) -> bool:
        """Check if schema has been analyzed and stored"""
        if not self.schema_status_file.exists():
            return False
        
        try:
            with open(self.schema_status_file, 'r', encoding='utf-8') as f:
                status = json.load(f)
                
            last_analyzed = datetime.fromisoformat(status.get('last_analyzed', '2000-01-01'))
            is_fresh = (datetime.now() - last_analyzed) < self.CACHE_DURATION
            db_matches = status.get('database') == self.conn_params['database']
            has_embeddings = self.schema_collection.count() > 0
            
            return is_fresh and db_matches and has_embeddings
        except Exception as e:
            logger.warning(f"Could not read schema status: {e}")
            return False

    async def analyze_complete_database(self, force_refresh: bool = False):
        """Perform comprehensive database analysis"""
        
        if not force_refresh and self.is_schema_analyzed():
            logger.info("✅ Schema already analyzed and cached. Loading from vector DB...")
            await self._load_schema_from_cache()
            return
        
        logger.info("🔄 Starting comprehensive database analysis...")
        
        conn = None
        cursor = None
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            
            await self._analyze_all_tables(cursor)
            await self._analyze_relationships(cursor)
            await self._sample_table_data(cursor)
            await self._build_schema_embeddings()
            await self._extract_business_patterns(cursor)
            
            self._last_refresh = datetime.now()
            
            await self._save_analysis_report()
            await self._save_schema_status()
            
            logger.info(f"✅ Database analysis complete: {len(self._schema_cache)} tables analyzed")
            
        except Exception as e:
            logger.error(f"Analysis failed: {e}")
            raise
        finally:
            if cursor:
                cursor.close()
            if conn:
                conn.close()

    async def _load_schema_from_cache(self):
        """Load schema information from cached analysis"""
        try:
            report_path = self.vector_db_path / "analysis_report.json"
            if report_path.exists():
                with open(report_path, 'r', encoding='utf-8') as f:
                    report = json.load(f)
                
                for table_name, table_data in report.get('tables', {}).items():
                    table_info = TableInfo(
                        table_name=table_name,
                        row_count=table_data.get('row_count', 0),
                        relationships=table_data.get('relationships', [])
                    )
                    
                    for col_data in table_data.get('columns', []):
                        col_info = ColumnInfo(
                            table_name=table_name,
                            column_name=col_data['name'],
                            data_type=col_data['type'],
                            semantic_type=col_data.get('semantic_type'),
                            is_primary=col_data.get('is_primary', False),
                            is_foreign=col_data.get('is_foreign', False)
                        )
                        table_info.columns.append(col_info)
                    
                    self._schema_cache[table_name] = table_info
                
                logger.info(f"📚 Loaded {len(self._schema_cache)} tables from cache")
        except Exception as e:
            logger.warning(f"Could not load schema from cache: {e}")

    async def _analyze_all_tables(self, cursor):
        """Analyze all tables in database"""
        try:
            cursor.execute("""
                SELECT 
                    TABLE_NAME,
                    TABLE_COMMENT,
                    TABLE_ROWS
                FROM information_schema.TABLES 
                WHERE TABLE_SCHEMA = %s 
                AND TABLE_TYPE = 'BASE TABLE'
                ORDER BY TABLE_NAME
            """, (self.conn_params['database'],))
            
            tables = cursor.fetchall()
            logger.info(f"Found {len(tables)} tables")
            
            for table_name, table_comment, row_count in tables:
                table_info = TableInfo(
                    table_name=table_name,
                    table_comment=table_comment,
                    row_count=row_count or 0
                )
                
                cursor.execute("""
                    SELECT 
                        COLUMN_NAME,
                        DATA_TYPE,
                        IS_NULLABLE,
                        COLUMN_KEY,
                        COLUMN_COMMENT,
                        CHARACTER_MAXIMUM_LENGTH,
                        NUMERIC_PRECISION,
                        NUMERIC_SCALE
                    FROM information_schema.COLUMNS 
                    WHERE TABLE_SCHEMA = %s 
                    AND TABLE_NAME = %s
                    ORDER BY ORDINAL_POSITION
                """, (self.conn_params['database'], table_name))
                
                for col_data in cursor.fetchall():
                    col_info = ColumnInfo(
                        table_name=table_name,
                        column_name=col_data[0],
                        data_type=self._format_data_type(col_data[1], col_data[5], col_data[6], col_data[7]),
                        is_nullable=(col_data[2] == 'YES'),
                        is_primary=(col_data[3] == 'PRI'),
                        column_comment=col_data[4],
                        semantic_type=self._detect_semantic_type(col_data[0], col_data[1])
                    )
                    table_info.columns.append(col_info)
                
                cursor.execute("""
                    SELECT DISTINCT INDEX_NAME
                    FROM information_schema.STATISTICS
                    WHERE TABLE_SCHEMA = %s 
                    AND TABLE_NAME = %s
                    AND INDEX_NAME != 'PRIMARY'
                """, (self.conn_params['database'], table_name))
                
                table_info.indexes = [row[0] for row in cursor.fetchall()]
                self._schema_cache[table_name] = table_info
                
        except Exception as e:
            logger.error(f"Failed to analyze tables: {e}")
            raise

    def _format_data_type(self, data_type: str, char_len, num_prec, num_scale) -> str:
        """Format data type with precision"""
        if char_len:
            return f"{data_type}({char_len})"
        elif num_prec:
            if num_scale:
                return f"{data_type}({num_prec},{num_scale})"
            return f"{data_type}({num_prec})"
        return data_type

    async def _analyze_relationships(self, cursor):
        """Analyze foreign key relationships"""
        try:
            cursor.execute("""
                SELECT 
                    kcu.TABLE_NAME,
                    kcu.COLUMN_NAME,
                    kcu.REFERENCED_TABLE_NAME,
                    kcu.REFERENCED_COLUMN_NAME,
                    kcu.CONSTRAINT_NAME
                FROM information_schema.KEY_COLUMN_USAGE kcu
                WHERE kcu.REFERENCED_TABLE_NAME IS NOT NULL
                AND kcu.TABLE_SCHEMA = %s
                ORDER BY kcu.TABLE_NAME, kcu.COLUMN_NAME
            """, (self.conn_params['database'],))
            
            for table, column, ref_table, ref_column, constraint in cursor.fetchall():
                if table in self._schema_cache:
                    relationship = {
                        'column': column,
                        'referenced_table': ref_table,
                        'referenced_column': ref_column,
                        'constraint_name': constraint
                    }
                    self._schema_cache[table].relationships.append(relationship)
                    
                    for col in self._schema_cache[table].columns:
                        if col.column_name == column:
                            col.is_foreign = True
                            col.foreign_ref = f"{ref_table}.{ref_column}"
        except Exception as e:
            logger.warning(f"Failed to analyze relationships: {e}")

    async def _sample_table_data(self, cursor):
        """Sample data from each table for better understanding"""
        for table_name, table_info in self._schema_cache.items():
            try:
                cursor.execute(f"SELECT * FROM `{table_name}` LIMIT 5")
                rows = cursor.fetchall()
                
                if rows:
                    for col_idx, col_info in enumerate(table_info.columns):
                        samples = []
                        for row in rows:
                            if row[col_idx] is not None:
                                samples.append(str(row[col_idx]))
                        col_info.sample_values = samples[:3]
                        
            except Exception as e:
                logger.warning(f"Could not sample data from {table_name}: {e}")

    def _detect_semantic_type(self, column_name: str, data_type: str) -> str:
        """Enhanced semantic type detection"""
        col_lower = column_name.lower()
        
        semantic_patterns = {
            'identifier': ['id', '_id', 'number', 'code', 'key', '_no', '_num'],
            'name': ['name', 'title', 'label', 'description', 'desc'],
            'email': ['email', 'mail'],
            'phone': ['phone', 'mobile', 'tel', 'contact'],
            'address': ['address', 'addr', 'street', 'city', 'state', 'country', 'zip', 'postal'],
            'amount': ['amount', 'price', 'cost', 'salary', 'total', 'sum', 'balance', 'credit', 'debit'],
            'date': ['date', 'time', 'created', 'updated', 'modified', 'timestamp'],
            'status': ['status', 'state', 'flag', 'active', 'enabled'],
            'percentage': ['percent', 'rate', 'ratio'],
            'url': ['url', 'link', 'website'],
        }
        
        for sem_type, patterns in semantic_patterns.items():
            if any(pattern in col_lower for pattern in patterns):
                return sem_type
        
        if 'char' in data_type.lower() or 'text' in data_type.lower():
            return 'text'
        elif 'int' in data_type.lower() or 'decimal' in data_type.lower() or 'float' in data_type.lower():
            return 'numeric'
        elif 'date' in data_type.lower() or 'time' in data_type.lower():
            return 'temporal'
        elif 'bool' in data_type.lower():
            return 'boolean'
        
        return 'unknown'

    async def _build_schema_embeddings(self):
        """Build vector embeddings for schema knowledge"""
        logger.info("Building schema embeddings...")
        
        documents = []
        metadatas = []
        ids = []
        
        for table_name, table_info in self._schema_cache.items():
            description = self._create_table_description(table_info)
            
            documents.append(description)
            metadatas.append({
                'type': 'table',
                'table_name': table_name,
                'row_count': str(table_info.row_count),
                'column_count': str(len(table_info.columns))
            })
            ids.append(f"table_{table_name}")
            
            for col in table_info.columns:
                col_description = self._create_column_description(col)
                
                documents.append(col_description)
                metadatas.append({
                    'type': 'column',
                    'table_name': table_name,
                    'column_name': col.column_name,
                    'data_type': col.data_type,
                    'semantic_type': col.semantic_type or 'unknown'
                })
                ids.append(f"column_{table_name}_{col.column_name}")
        
        if documents:
            try:
                self.schema_collection.upsert(
                    documents=documents,
                    metadatas=metadatas,
                    ids=ids
                )
                logger.info(f"✅ Created {len(documents)} schema embeddings")
            except Exception as e:
                logger.error(f"Failed to create embeddings: {e}")
                raise

    def _create_table_description(self, table_info: TableInfo) -> str:
        """Create rich natural language description of table"""
        desc = f"Table: {table_info.table_name}"
        
        if table_info.table_comment:
            desc += f". Description: {table_info.table_comment}"
        
        desc += f". Contains {len(table_info.columns)} columns"
        
        if table_info.row_count:
            desc += f" with approximately {table_info.row_count} records"
        
        key_cols = [col.column_name for col in table_info.columns if col.is_primary]
        if key_cols:
            desc += f". Primary key: {', '.join(key_cols)}"
        
        if table_info.relationships:
            rel_desc = [f"{r['column']} references {r['referenced_table']}" 
                       for r in table_info.relationships[:3]]
            desc += f". Relationships: {'; '.join(rel_desc)}"
        
        return desc

    def _create_column_description(self, col: ColumnInfo) -> str:
        """Create natural language description of column"""
        desc = f"Column {col.column_name} in table {col.table_name}"
        desc += f". Data type: {col.data_type}"
        desc += f". Semantic type: {col.semantic_type or 'general'}"
        
        if col.is_primary:
            desc += ". This is a primary key"
        
        if col.is_foreign and col.foreign_ref:
            desc += f". Foreign key referencing {col.foreign_ref}"
        
        if col.column_comment:
            desc += f". Description: {col.column_comment}"
        
        if col.sample_values:
            desc += f". Example values: {', '.join(col.sample_values[:3])}"
        
        return desc

    async def _extract_business_patterns(self, cursor):
        """Extract business logic patterns from data"""
        logger.info("Extracting business patterns...")
        
        patterns = []
        
        for table_name, table_info in self._schema_cache.items():
            if 'customer' in table_name.lower():
                patterns.append(f"The {table_name} table stores customer information")
            elif 'order' in table_name.lower():
                patterns.append(f"The {table_name} table tracks orders or transactions")
            elif 'product' in table_name.lower():
                patterns.append(f"The {table_name} table contains product catalog")
            elif 'employee' in table_name.lower():
                patterns.append(f"The {table_name} table manages employee records")
            
            amount_cols = [col for col in table_info.columns if col.semantic_type == 'amount']
            if amount_cols:
                for col in amount_cols:
                    patterns.append(
                        f"To calculate total {col.column_name} in {table_name}, use SUM({col.column_name})"
                    )
        
        if patterns:
            try:
                self.business_collection.upsert(
                    documents=patterns,
                    metadatas=[{'type': 'pattern'} for _ in patterns],
                    ids=[f"pattern_{i}" for i in range(len(patterns))]
                )
            except Exception as e:
                logger.warning(f"Failed to store business patterns: {e}")

    async def _save_analysis_report(self):
        """Save comprehensive analysis report"""
        try:
            report_path = self.vector_db_path / "analysis_report.json"
            
            report = {
                'analyzed_at': str(datetime.now()),
                'database': self.conn_params['database'],
                'total_tables': len(self._schema_cache),
                'tables': {}
            }
            
            for table_name, table_info in self._schema_cache.items():
                report['tables'][table_name] = {
                    'row_count': table_info.row_count,
                    'column_count': len(table_info.columns),
                    'columns': [
                        {
                            'name': col.column_name,
                            'type': col.data_type,
                            'semantic_type': col.semantic_type,
                            'is_primary': col.is_primary,
                            'is_foreign': col.is_foreign
                        }
                        for col in table_info.columns
                    ],
                    'relationships': table_info.relationships
                }
            
            with open(report_path, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
            
            logger.info(f"📄 Analysis report saved to {report_path}")
        except Exception as e:
            logger.error(f"Failed to save analysis report: {e}")

    async def _save_schema_status(self):
        """Save schema analysis status"""
        try:
            status = {
                'last_analyzed': datetime.now().isoformat(),
                'database': self.conn_params['database'],
                'tables_count': len(self._schema_cache),
                'embeddings_count': self.schema_collection.count()
            }
            
            with open(self.schema_status_file, 'w', encoding='utf-8') as f:
                json.dump(status, f, indent=2, ensure_ascii=False)
            
            logger.info("✅ Schema status saved")
        except Exception as e:
            logger.error(f"Failed to save schema status: {e}")

    def get_schema_summary(self) -> Dict[str, Any]:
        """Get human-readable schema summary"""
        return {
            'database': self.conn_params['database'],
            'total_tables': len(self._schema_cache),
            'tables': [
                {
                    'name': name,
                    'rows': info.row_count,
                    'columns': len(info.columns),
                    'has_relationships': len(info.relationships) > 0
                }
                for name, info in sorted(self._schema_cache.items())
            ]
        }
    
    def get_complete_schema_text(self) -> str:
        """Generate complete schema as text for LLM understanding"""
        schema_text = f"DATABASE: {self.conn_params['database']}\n"
        schema_text += "=" * 80 + "\n\n"
        
        for table_name, table_info in sorted(self._schema_cache.items()):
            schema_text += f"TABLE: {table_name}\n"
            if table_info.table_comment:
                schema_text += f"Description: {table_info.table_comment}\n"
            schema_text += f"Rows: ~{table_info.row_count}\n"
            schema_text += "-" * 40 + "\n"
            
            schema_text += "COLUMNS:\n"
            for col in table_info.columns:
                col_line = f"  - {col.column_name} ({col.data_type})"
                if col.is_primary:
                    col_line += " [PRIMARY KEY]"
                if col.is_foreign:
                    col_line += f" [FOREIGN KEY -> {col.foreign_ref}]"
                if col.semantic_type:
                    col_line += f" [Type: {col.semantic_type}]"
                if col.column_comment:
                    col_line += f"\n    Description: {col.column_comment}"
                if col.sample_values:
                    col_line += f"\n    Examples: {', '.join(col.sample_values)}"
                schema_text += col_line + "\n"
            
            if table_info.relationships:
                schema_text += "\nRELATIONSHIPS:\n"
                for rel in table_info.relationships:
                    schema_text += f"  - {rel['column']} -> {rel['referenced_table']}.{rel['referenced_column']}\n"
            
            schema_text += "\n"
        
        return schema_text

class IntelligentQueryResolver:
    """Advanced query resolution using RAG"""
    
    def __init__(self, analyzer: EnhancedDatabaseAnalyzer):
        self.analyzer = analyzer
        
    async def resolve_query(self, natural_query: str) -> Dict[str, Any]:
        """Resolve natural language query using RAG"""
        try:
            schema_results = self.analyzer.schema_collection.query(
                query_texts=[natural_query],
                n_results=5
            )
            
            history_results = self.analyzer.query_collection.query(
                query_texts=[natural_query],
                n_results=3
            )
            
            business_results = self.analyzer.business_collection.query(
                query_texts=[natural_query],
                n_results=3
            )
            
            return {
                'schema_context': self._process_schema_results(schema_results),
                'similar_queries': self._process_history_results(history_results),
                'business_context': self._process_business_results(business_results)
            }
        except Exception as e:
            logger.error(f"Query resolution failed: {e}")
            return {
                'schema_context': [],
                'similar_queries': [],
                'business_context': []
            }
    
    def _process_schema_results(self, results) -> List[Dict]:
        """Process schema search results"""
        if not results['documents'] or not results['documents'][0]:
            return []
        
        processed = []
        for doc, metadata, distance in zip(
            results['documents'][0],
            results['metadatas'][0],
            results['distances'][0]
        ):
            processed.append({
                'content': doc,
                'metadata': metadata,
                'relevance': 1 - distance
            })
        
        return processed
    
    def _process_history_results(self, results) -> List[Dict]:
        """Process historical query results"""
        if not results['documents'] or not results['documents'][0]:
            return []
        
        processed = []
        for doc, metadata, distance in zip(
            results['documents'][0],
            results['metadatas'][0],
            results['distances'][0]
        ):
            if distance < 0.3:
                processed.append({
                    'query': doc,
                    'sql': metadata.get('sql', ''),
                    'result_path': metadata.get('result_path'),
                    'timestamp': metadata.get('timestamp'),
                    'relevance': 1 - distance
                })
        
        return processed
    
    def _process_business_results(self, results) -> List[Dict]:
        """Process business logic results"""
        if not results['documents'] or not results['documents'][0]:
            return []
        
        return [doc for doc in results['documents'][0]]

class AdvancedTextToSQLEngine:
    """Production-grade Text-to-SQL engine with RAG and Ollama LLM"""
    
    def __init__(self, connection_params: dict, 
                 ollama_model: str = "qwen2.5-coder:7b",
                 ollama_host: str = "http://127.0.0.1:11434",
                 vector_db_path: str = "./vector_db",
                 output_dir: str = "./query_outputs"):
        
        self.connection_params = connection_params
        self.analyzer = EnhancedDatabaseAnalyzer(connection_params, vector_db_path)
        self.resolver = IntelligentQueryResolver(self.analyzer)
        
        # Initialize Ollama
        self.ollama_model = ollama_model
        self.ollama_host = ollama_host
        self.ollama_client = ollama.Client(host=ollama_host)
        
        # Verify Ollama connection and model
        self._verify_ollama_setup()
        
        # Output directory for CSV and JSON files
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # LLM schema understanding cache
        self.llm_understanding_file = Path(vector_db_path) / "llm_schema_understanding.json"
        self.schema_understood_by_llm = False
        
        # Query result cache
        self.cache_dir = Path(vector_db_path) / "query_cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_cache_files = 200
        
        logger.info(f"✅ Engine initialized with Ollama model: {self.ollama_model}")

    def _verify_ollama_setup(self):
        """Verify Ollama is running and model is available"""
        # Diagnostic: print the resolved base URL used by the underlying HTTP client
        try:
            base_url = getattr(self.ollama_client._client, 'base_url', None)
            logger.info(f"🔍 Ollama client base_url = {base_url}")

            # If the client resolved to 0.0.0.0 (listen-all), many HTTP clients cannot reach that
            # address. Recreate the client to point to loopback (127.0.0.1) which is reachable.
            if base_url and '0.0.0.0' in str(base_url):
                logger.info("⚠️ Detected Ollama base_url bound to 0.0.0.0 — recreating client to use 127.0.0.1")
                # Prefer existing OLLAMA_HOST if it contains 127.0.0.1 already, otherwise set it.
                try:
                    # Update host string safely
                    new_host = 'http://127.0.0.1:11434'
                    self.ollama_host = new_host
                    self.ollama_client = ollama.Client(host=new_host)
                    base_url = getattr(self.ollama_client._client, 'base_url', None)
                    logger.info(f"🔍 New Ollama client base_url = {base_url}")
                except Exception as e:
                    logger.warning(f"Failed to recreate Ollama client for 127.0.0.1: {e}")
        except Exception:
            logger.info(f"🔍 Ollama client host (raw): {self.ollama_host}")

        # Try a small HTTP probe with retries to give clearer diagnostics
        import httpx

        attempts = 4
        backoff = 0.5
        last_exc = None

        for attempt in range(1, attempts + 1):

            try:
                # Use the client list call which is the canonical check
                models = self.ollama_client.list()
                # Log the first model for debugging
                if models.get('models') and len(models['models']) > 0:
                    logger.debug(f"First Ollama model object: {models['models'][0]}")
                # Safely access 'name' key
                available_models = [model['name'] if isinstance(model, dict) and 'name' in model else model.get('model', '<unknown>') for model in models['models']]

                logger.info(f"✅ Connected to Ollama at {self.ollama_host}")
                logger.info(f"📋 Available models: {', '.join(available_models)}")

                # Check if requested model is available
                model_base = self.ollama_model.split(':')[0]
                if not any(model_base in m for m in available_models):
                    logger.warning(f"⚠️ Model '{self.ollama_model}' not found. Attempting to pull...")
                    try:
                        self.ollama_client.pull(self.ollama_model)
                        logger.info(f"✅ Model '{self.ollama_model}' pulled successfully")
                    except Exception as e:
                        logger.error(f"❌ Failed to pull model: {e}")
                        logger.info("💡 Available models for SQL tasks:")
                        logger.info("   - qwen2.5-coder:7b (Recommended for SQL)")
                        logger.info("   - codellama:7b")
                        logger.info("   - mistral:7b")
                        logger.info("   - llama3.1:8b")
                        raise Exception(f"Model {self.ollama_model} not available. Please pull it using: ollama pull {self.ollama_model}")

                return

            except Exception as e:
                last_exc = e
                # If it's an httpx ConnectError, give a slightly clearer message
                err_name = type(e).__name__
                logger.warning(f"Attempt {attempt}/{attempts}: Could not reach Ollama ({err_name}): {e}")

                # If this was a connection problem, try a low-level http probe for extra info
                try:
                    probe_client = httpx.Client(base_url=base_url, timeout=2.0)
                    resp = probe_client.get('/api/tags')
                    logger.info(f"Probe HTTP status: {resp.status_code}")
                    try:
                        logger.info(f"Probe response snippet: {resp.text[:200]}")
                    except Exception:
                        pass
                except Exception as probe_exc:
                    logger.debug(f"Probe attempt failed: {probe_exc}")

                if attempt < attempts:
                    import time
                    time.sleep(backoff * attempt)

        logger.error(f"❌ Failed to connect to Ollama: {last_exc}")
        logger.info("💡 Make sure Ollama is running: ollama serve")
        raise last_exc

    def _save_results_to_csv(self, results: List[Dict], query_id: str, query: str) -> str:
        """Save query results to CSV file with proper error handling"""
        try:
            if not results:
                logger.warning("No results to save to CSV")
                return None
            
            # Create safe filename from query
            safe_query = re.sub(r'[^\w\s-]', '', query)[:50]
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"{timestamp}_{safe_query}_{query_id[:8]}.csv"
            filepath = self.output_dir / filename
            
            # Convert to DataFrame for better CSV handling
            df = pd.DataFrame(results)
            
            # Save with proper encoding and error handling
            df.to_csv(filepath, index=False, encoding='utf-8-sig', quoting=csv.QUOTE_ALL)
            
            logger.info(f"✅ Results saved to CSV: {filepath}")
            return str(filepath)
            
        except Exception as e:
            logger.error(f"Failed to save CSV: {e}")
            # Fallback: Try manual CSV writing
            try:
                with open(filepath, 'w', newline='', encoding='utf-8-sig') as f:
                    if results:
                        writer = csv.DictWriter(f, fieldnames=results[0].keys(), quoting=csv.QUOTE_ALL)
                        writer.writeheader()
                        writer.writerows(results)
                logger.info(f"✅ Results saved to CSV (fallback method): {filepath}")
                return str(filepath)
            except Exception as e2:
                logger.error(f"CSV fallback also failed: {e2}")
                return None

    def _save_results_to_json(self, results: List[Dict], query_id: str, query: str, metadata: Dict = None) -> str:
        """Save query results to JSON file with metadata"""
        try:
            if not results:
                logger.warning("No results to save to JSON")
                return None
            
            safe_query = re.sub(r'[^\w\s-]', '', query)[:50]
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"{timestamp}_{safe_query}_{query_id[:8]}.json"
            filepath = self.output_dir / filename
            
            output_data = {
                'query': query,
                'timestamp': datetime.now().isoformat(),
                'row_count': len(results),
                'results': results
            }
            
            if metadata:
                output_data['metadata'] = metadata
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, indent=2, default=str, ensure_ascii=False)
            
            logger.info(f"✅ Results saved to JSON: {filepath}")
            return str(filepath)
            
        except Exception as e:
            logger.error(f"Failed to save JSON: {e}")
            return None

    def stream_natural_response_sync(self, query: str, result: Dict[str, Any]):
        """Synchronous streaming natural response using Ollama"""
        if result['row_count'] == 0:
            print("No results found for your query.")
            return

        data_summary = json.dumps(result['data'][:10], indent=2, default=str)

        prompt = f"""Convert this SQL query result into a natural, conversational response.

Original question: {query}
Number of results: {result['row_count']}
Sample data:
{data_summary}

Generate a clear, concise response that:
1. Answers the question directly
2. Mentions key findings from the data
3. Uses natural language (no technical jargon)
4. Return the full answer. Do not end with 'and N more'. If the result set is large, summarize key insights.

Response:"""

        try:
            stream = self.ollama_client.chat(
                model=self.ollama_model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that explains data clearly."},
                    {"role": "user", "content": prompt}
                ],
                stream=True
            )
            
            for chunk in stream:
                if 'message' in chunk and 'content' in chunk['message']:
                    print(chunk['message']['content'], end='', flush=True)
            
            print()  # New line after streaming
            
        except Exception as e:
            logger.warning(f"Streaming failed: {e}, using fallback")
            text = asyncio.get_event_loop().run_until_complete(self._generate_natural_response(query, result))
            print(text)

    async def stream_natural_response_async(self, query: str, result: Dict[str, Any]):
        """Async wrapper for streaming natural response"""
        await asyncio.to_thread(self.stream_natural_response_sync, query, result)
    
    async def initialize(self, force_schema_refresh: bool = False):
        """Initialize the engine and ensure LLM understands schema"""
        logger.info("🔄 Initializing engine...")
        
        try:
            await self.analyzer.analyze_complete_database(force_refresh=force_schema_refresh)
            
            if not force_schema_refresh and self._check_llm_understanding():
                logger.info("✅ LLM schema understanding loaded from cache")
                self.schema_understood_by_llm = True
            else:
                logger.info("🧠 Teaching LLM about database schema...")
                await self._teach_llm_schema()
                self.schema_understood_by_llm = True
            
            logger.info("✅ Engine ready with full schema understanding!")
        except Exception as e:
            logger.error(f"Initialization failed: {e}")
            raise
    
    def _check_llm_understanding(self) -> bool:
        """Check if LLM understanding is cached"""
        if not self.llm_understanding_file.exists():
            return False
        
        try:
            with open(self.llm_understanding_file, 'r', encoding='utf-8') as f:
                understanding = json.load(f)
            
            db_matches = understanding.get('database') == self.connection_params['database']
            tables_match = understanding.get('tables_count') == len(self.analyzer._schema_cache)
            
            last_understood = datetime.fromisoformat(understanding.get('last_understood', '2000-01-01'))
            is_fresh = (datetime.now() - last_understood) < timedelta(hours=6)
            
            return db_matches and tables_match and is_fresh
        except Exception as e:
            logger.warning(f"Could not load LLM understanding: {e}")
            return False
    
    async def _teach_llm_schema(self):
        """Teach the LLM about the complete database schema"""
        try:
            schema_text = self.analyzer.get_complete_schema_text()
            
            learning_prompt = f"""You are a database expert assistant. Study and understand this complete database schema:

{schema_text}

Your task is to understand:
1. All table structures and their purposes
2. Relationships between tables (foreign keys)
3. Data types and constraints
4. Sample data patterns
5. Business logic implied by table and column names

Please provide a comprehensive summary of your understanding of this database, including:
- Main entities and their relationships
- Key business concepts represented
- Important patterns for common queries
- Any insights about the data model

Summary:"""

            response = self.ollama_client.chat(
                model=self.ollama_model,
                messages=[
                    {"role": "system", "content": "You are a database expert who analyzes and understands database schemas deeply."},
                    {"role": "user", "content": learning_prompt}
                ],
                options={
                    "temperature": 0.3,
                    "num_predict": 2000
                }
            )
            
            llm_understanding = response['message']['content'].strip()
            
            understanding_id = f"llm_understanding_{self.connection_params['database']}"
            self.analyzer.llm_schema_collection.upsert(
                documents=[llm_understanding],
                metadatas=[{
                    'type': 'llm_understanding',
                    'database': self.connection_params['database'],
                    'timestamp': str(datetime.now()),
                    'full_schema': schema_text[:10000]
                }],
                ids=[understanding_id]
            )
            
            understanding_data = {
                'database': self.connection_params['database'],
                'tables_count': len(self.analyzer._schema_cache),
                'last_understood': datetime.now().isoformat(),
                'llm_summary': llm_understanding,
                'schema_hash': hashlib.md5(schema_text.encode()).hexdigest()
            }
            
            with open(self.llm_understanding_file, 'w', encoding='utf-8') as f:
                json.dump(understanding_data, f, indent=2, ensure_ascii=False)
            
            logger.info("✅ LLM has learned and understood the complete database schema")
            logger.info(f"📝 Understanding summary:\n{llm_understanding[:500]}...")
            
        except Exception as e:
            logger.error(f"Failed to teach LLM schema: {e}")
    
    async def process_query(self, natural_query: str, save_to_csv: bool = True, save_to_json: bool = False) -> Dict[str, Any]:
        """Process natural language query end-to-end with file saving options"""
        start_time = datetime.now()
        query_id = hashlib.md5(natural_query.encode()).hexdigest()
        
        try:
            if self._is_meta_query(natural_query):
                return await self._handle_meta_query(natural_query)
            
            context = await self.resolver.resolve_query(natural_query)
            
            # Check cache
            if context['similar_queries'] and context['similar_queries'][0]['relevance'] > 0.95:
                logger.info("♻️ Using cached query")
                cached_query = context['similar_queries'][0]
                result_path = cached_query.get('result_path')
                
                if result_path and Path(result_path).exists():
                    try:
                        with open(result_path, 'r', encoding='utf-8') as rf:
                            cached_data = json.load(rf)
                        
                        nl_response = cached_data.get('natural_response') or await self._generate_natural_response(natural_query, cached_data)
                        
                        return {
                            'success': True,
                            'natural_query': natural_query,
                            'sql': cached_query.get('sql', ''),
                            'results': cached_data['data'],
                            'row_count': cached_data['row_count'],
                            'natural_response': nl_response,
                            'execution_time': float(cached_data.get('execution_time', 0.0)),
                            'from_cache': True,
                            'csv_path': cached_data.get('csv_path'),
                            'json_path': cached_data.get('json_path')
                        }
                    except Exception as e:
                        logger.warning(f"Failed to load cached result: {e}")
                
                exec_result = await self._execute_sql(cached_query.get('sql', 'SELECT 1'))
                nl_response = await self._generate_natural_response(natural_query, exec_result)
                
                return {
                    'success': True,
                    'natural_query': natural_query,
                    'sql': cached_query.get('sql', ''),
                    'results': exec_result['data'],
                    'row_count': exec_result['row_count'],
                    'natural_response': nl_response,
                    'execution_time': exec_result.get('execution_time', 0.0),
                    'from_cache': True
                }
            
            # Generate SQL
            sql = await self._generate_sql_with_context(natural_query, context)
            
            # Execute query
            result = await self._execute_sql(sql)
            
            # Save to files if requested
            csv_path = None
            json_path = None
            
            if save_to_csv and result['data']:
                csv_path = self._save_results_to_csv(result['data'], query_id, natural_query)
            
            if save_to_json and result['data']:
                json_path = self._save_results_to_json(
                    result['data'], 
                    query_id, 
                    natural_query,
                    metadata={
                        'sql': sql,
                        'execution_time': result['execution_time']
                    }
                )
            
            # Generate natural language response
            nl_response = await self._generate_natural_response(natural_query, result)
            
            # Cache the successful query
            await self._cache_query(natural_query, sql, result, nl_response, csv_path, json_path)
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            return {
                'success': True,
                'natural_query': natural_query,
                'sql': sql,
                'results': result['data'],
                'row_count': result['row_count'],
                'natural_response': nl_response,
                'execution_time': execution_time,
                'from_cache': False,
                'csv_path': csv_path,
                'json_path': json_path
            }
            
        except Exception as e:
            logger.error(f"Query processing failed: {e}")
            return {
                'success': False,
                'natural_query': natural_query,
                'error': str(e)
            }
    
    def _is_meta_query(self, query: str) -> bool:
        """Check if query is about database structure itself"""
        meta_patterns = [
            r'how many tables',
            r'list (?:all )?tables',
            r'show (?:all )?tables',
            r'what tables',
            r'database structure',
            r'schema',
            r'table names',
            r'describe (?:all )?tables?',
            r'list columns'
        ]
        
        query_lower = query.lower()
        return any(re.search(pattern, query_lower) for pattern in meta_patterns)
    
    async def _handle_meta_query(self, query: str) -> Dict[str, Any]:
        """Handle queries about database structure"""
        query_lower = query.lower()
        
        if 'how many tables' in query_lower:
            count = len(self.analyzer._schema_cache)
            table_names = list(self.analyzer._schema_cache.keys())
            
            return {
                'success': True,
                'natural_query': query,
                'sql': 'SHOW TABLES',
                'results': [{'table': name} for name in table_names],
                'row_count': count,
                'natural_response': f"Your database contains {count} tables: {', '.join(table_names)}",
                'execution_time': 0.001,
                'from_cache': False
            }
        
        elif any(kw in query_lower for kw in ['list tables', 'show tables', 'what tables']):
            summary = self.analyzer.get_schema_summary()
            
            response = f"Your database '{summary['database']}' has {summary['total_tables']} tables:\n\n"
            for table in summary['tables']:
                response += f"• {table['name']}: {table['rows']} rows, {table['columns']} columns\n"
            
            return {
                'success': True,
                'natural_query': query,
                'sql': 'SHOW TABLES',
                'results': summary['tables'],
                'row_count': summary['total_tables'],
                'natural_response': response,
                'execution_time': 0.001,
                'from_cache': False
            }
        
        return await self.process_query(query)
    
    async def _generate_sql_with_context(self, query: str, context: Dict) -> str:
        """Generate SQL using Ollama with RAG context and schema understanding"""
        
        schema_context = "\n".join([
            f"- {item['content']}"
            for item in context['schema_context'][:5]
        ])
        
        business_context = "\n".join([
            f"- {item}"
            for item in context['business_context'][:3]
        ])
        
        similar_examples = ""
        if context['similar_queries']:
            similar_examples = "\nSimilar queries:\n"
            for sq in context['similar_queries'][:2]:
                similar_examples += f"Query: {sq['query']}\nSQL: {sq['sql']}\n\n"
        
        llm_context = ""
        if self.schema_understood_by_llm:
            try:
                understanding_results = self.analyzer.llm_schema_collection.query(
                    query_texts=[query],
                    n_results=1
                )
                if understanding_results['documents'] and understanding_results['documents'][0]:
                    llm_context = f"\n\nYOUR SCHEMA UNDERSTANDING:\n{understanding_results['documents'][0][0]}\n"
            except:
                pass
        
        prompt = f"""You are an expert SQL generator for MySQL databases. You have thoroughly studied this database schema.

RELEVANT SCHEMA CONTEXT:
{schema_context}

BUSINESS LOGIC:
{business_context}

{llm_context}

{similar_examples}

USER QUERY: {query}

Generate a valid MySQL SELECT query that answers the user's question.

RULES:
1. Only generate SELECT statements (no INSERT, UPDATE, DELETE, DROP)
2. Use proper table and column names from the context
3. Use JOINs when referencing multiple tables
4. Add appropriate WHERE, GROUP BY, ORDER BY, and LIMIT clauses
5. For safety, always add LIMIT clause (max 1000)
6. Return ONLY the SQL query, no explanations

SQL:"""
        
        attempts = 3
        backoff = 0.5
        last_exc = None
        
        for attempt in range(attempts):
            try:
                response = self.ollama_client.chat(
                    model=self.ollama_model,
                    messages=[
                        {"role": "system", "content": "You are a SQL expert. Generate only valid MySQL queries."},
                        {"role": "user", "content": prompt}
                    ],
                    options={
                        "temperature": 0.1,
                        "num_predict": 500
                    }
                )

                sql = response['message']['content'].strip()
                sql = self._clean_sql(sql)
                sql = self._validate_sql(sql)
                return sql
            except Exception as e:
                last_exc = e
                logger.warning(f"LLM generation attempt {attempt+1} failed: {e}")
                await asyncio.sleep(backoff * (attempt + 1))

        logger.error(f"LLM SQL generation failed after {attempts} attempts: {last_exc}")
        raise last_exc
    
    def _clean_sql(self, sql: str) -> str:
        """Clean and format SQL"""
        sql = sql.replace('```sql', '').replace('```', '').strip()
        sql = sql.rstrip(';')
        return sql
    
    def _validate_sql(self, sql: str) -> str:
        """Validate and sanitize SQL"""
        if not sql:
            raise ValueError("Empty SQL query generated")
        
        dangerous_patterns = [
            r'\b(DROP|DELETE|UPDATE|INSERT|CREATE|ALTER|TRUNCATE|EXEC|EXECUTE|GRANT|REVOKE)\b'
        ]
        
        for pattern in dangerous_patterns:
            if re.search(pattern, sql, re.IGNORECASE):
                raise ValueError(f"Dangerous SQL operation detected")
        
        if not sql.strip().upper().startswith('SELECT'):
            raise ValueError("Only SELECT statements allowed")
        
        if 'LIMIT' not in sql.upper():
            sql += ' LIMIT 100'
        
        return sql
    
    async def _execute_sql(self, sql: str) -> Dict[str, Any]:
        """Execute SQL and return results"""
        start_time = datetime.now()
        conn = None
        cursor = None
        
        try:
            conn = self.analyzer.get_connection()
            cursor = conn.cursor(pymysql.cursors.DictCursor)
            
            cursor.execute(sql)
            results = cursor.fetchall()
            execution_time = (datetime.now() - start_time).total_seconds()
            
            return {
                'data': results,
                'row_count': len(results),
                'execution_time': execution_time
            }
        except Exception as e:
            logger.error(f"SQL execution failed: {e}")
            raise
        finally:
            if cursor:
                cursor.close()
            if conn:
                conn.close()
    
    async def _generate_natural_response(self, query: str, result: Dict) -> str:
        """Generate natural language response from SQL results"""
        
        if result['row_count'] == 0:
            return "No results found for your query."
        
        data_summary = json.dumps(result['data'][:3], indent=2, default=str)
        
        prompt = f"""Convert this SQL query result into a natural, conversational response.

Original question: {query}
Number of results: {result['row_count']}
Sample data:
{data_summary}

Generate a clear, concise response that:
1. Answers the question directly
2. Mentions key findings from the data
3. Uses natural language (no technical jargon)
4. Is friendly and professional

IMPORTANT: Return the full answer. Do not end with phrases like '... and N more'. If the result set is large, summarize key insights.

Response:"""

        attempts = 3
        backoff = 0.5
        last_exc = None
        
        for attempt in range(attempts):
            try:
                response = self.ollama_client.chat(
                    model=self.ollama_model,
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant that explains data clearly."},
                        {"role": "user", "content": prompt}
                    ],
                    options={
                        "temperature": 0.7,
                        "num_predict": 600
                    }
                )

                text = response['message']['content'].strip()
                text = re.sub(r"\*\*?\d+ more\*\*?", "", text)
                text = re.sub(r"and \d+ more", "", text)

                return text
            except Exception as e:
                last_exc = e
                logger.warning(f"Natural response attempt {attempt+1} failed: {e}")
                await asyncio.sleep(backoff * (attempt + 1))

        logger.warning(f"Natural response generation failed: {last_exc}")
        return f"Found {result['row_count']} results."
    
    async def _cache_query(self, query: str, sql: str, result: Dict, natural_response: str = None, 
                          csv_path: str = None, json_path: str = None):
        """Cache successful query for future use"""
        try:
            query_id = hashlib.md5(query.encode()).hexdigest()
            cached = dict(result)
            
            if natural_response is not None:
                cached['natural_response'] = natural_response
            if csv_path:
                cached['csv_path'] = csv_path
            if json_path:
                cached['json_path'] = json_path

            result_path = str(self.cache_dir / f"{query_id}.json")
            with open(result_path, 'w', encoding='utf-8') as wf:
                json.dump(cached, wf, default=str, ensure_ascii=False, indent=2)

            metadata = {
                'sql': sql,
                'timestamp': str(datetime.now()),
                'row_count': str(result['row_count']),
                'execution_time': str(result['execution_time']),
                'result_path': result_path,
                'nl_cached': bool(natural_response is not None)
            }

            self.analyzer.query_collection.upsert(
                documents=[query],
                metadatas=[metadata],
                ids=[query_id]
            )

            # Evict old cache files
            cache_files = sorted(self.cache_dir.glob('*.json'), key=lambda p: p.stat().st_mtime)
            if len(cache_files) > self.max_cache_files:
                to_remove = cache_files[: len(cache_files) - self.max_cache_files]
                for p in to_remove:
                    try:
                        p.unlink()
                    except Exception:
                        pass
        except Exception as e:
            logger.warning(f"Query caching failed: {e}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get engine statistics"""
        try:
            return {
                'cached_queries': self.analyzer.query_collection.count(),
                'schema_embeddings': self.analyzer.schema_collection.count(),
                'business_patterns': self.analyzer.business_collection.count(),
                'tables_analyzed': len(self.analyzer._schema_cache),
                'llm_understood': self.schema_understood_by_llm,
                'llm_understanding_stored': self.analyzer.llm_schema_collection.count() > 0,
                'ollama_model': self.ollama_model,
                'output_directory': str(self.output_dir)
            }
        except Exception as e:
            logger.error(f"Failed to get statistics: {e}")
            return {}
    
    def display_schema_understanding(self):
        """Display LLM's understanding of the schema"""
        if not self.llm_understanding_file.exists():
            print("❌ LLM has not yet learned the schema")
            return
        
        try:
            with open(self.llm_understanding_file, 'r', encoding='utf-8') as f:
                understanding = json.load(f)
            
            print("\n" + "=" * 80)
            print("🧠 LLM SCHEMA UNDERSTANDING")
            print("=" * 80)
            print(f"Database: {understanding.get('database')}")
            print(f"Tables Analyzed: {understanding.get('tables_count')}")
            print(f"Last Updated: {understanding.get('last_understood')}")
            print("\n" + "-" * 80)
            print("Summary:")
            print("-" * 80)
            print(understanding.get('llm_summary', 'No summary available'))
            print("=" * 80 + "\n")
        except Exception as e:
            logger.error(f"Could not display schema understanding: {e}")


async def main():
    """Example usage and testing"""
    
    # Load configuration from environment variables
    connection_params = {
        "host": os.getenv("DB_HOST"),
        "user": os.getenv("DB_USER"),
        "password": os.getenv("DB_PASSWORD"),
        "database": os.getenv("DB_NAME", "chinook"),
        "port": int(os.getenv("DB_PORT", "3306"))
    }
    
    # Initialize engine with Ollama
    engine = AdvancedTextToSQLEngine(
        connection_params=connection_params,
        ollama_model=os.getenv("OLLAMA_MODEL"),
        ollama_host=os.getenv("OLLAMA_HOST"),
        vector_db_path=os.getenv("VECTOR_DB_PATH"),
        output_dir=os.getenv("OUTPUT_DIR")
    )
    
    print("=" * 80)
    print("🚀 Advanced Text-to-SQL Engine with RAG + Ollama LLM")
    print("=" * 80)
    print("\n⏳ Analyzing database and teaching LLM...")
    print("💡 On subsequent runs, cached schema understanding will be loaded instantly!\n")
    
    await engine.initialize(force_schema_refresh=False)
    
    print("\n✅ Database analysis and LLM learning complete!")
    
    engine.display_schema_understanding()
    
    stats = engine.get_statistics()
    print("\n📊 Engine Statistics:")
    for key, value in stats.items():
        formatted_key = key.replace('_', ' ').title()
        print(f"   • {formatted_key}: {value}")
    
    # Interactive mode
    print("\n🎯 Interactive Mode")
    print("💡 The LLM now fully understands your database schema!")
    print("💡 Commands: 'schema' | 'stats' | 'exit' | ask any question")
    print("-" * 80)
    
    while True:
        try:
            user_query = input("\n💭 Your question: ").strip()
            
            if not user_query:
                continue
            
            if user_query.lower() in ['exit', 'quit', 'q']:
                print("\n👋 Goodbye!")
                break
            
            if user_query.lower() == 'schema':
                engine.display_schema_understanding()
                continue
            
            if user_query.lower() == 'stats':
                stats = engine.get_statistics()
                print("\n📊 Current Statistics:")
                for key, value in stats.items():
                    formatted_key = key.replace('_', ' ').title()
                    print(f"   • {formatted_key}: {value}")
                continue
            
            print("\n⏳ Processing...")
            result = await engine.process_query(user_query, save_to_csv=True, save_to_json=False)

            if result['success']:
                cache_indicator = "♻️" if result.get('from_cache') else "🆕"
                print(f"\n{cache_indicator} SQL: {result['sql']}")
                print(f"📊 Found {result['row_count']} results in {result['execution_time']:.3f}s")
                
                if result.get('csv_path'):
                    print(f"💾 CSV saved to: {result['csv_path']}")
                if result.get('json_path'):
                    print(f"💾 JSON saved to: {result['json_path']}")

                print("\n💬 Response:")
                await engine.stream_natural_response_async(user_query, {
                    'data': result['results'],
                    'row_count': result['row_count'],
                    'execution_time': result['execution_time']
                })

                MAX_PRINT_ROWS = 10
                if result['row_count'] > 0 and result['row_count'] <= MAX_PRINT_ROWS:
                    print(f"\n📋 Data:")
                    for idx, row in enumerate(result['results'], 1):
                        print(f"   {idx}. {row}")
                elif result['row_count'] > MAX_PRINT_ROWS:
                    print(f"\n📋 Sample Data (first {MAX_PRINT_ROWS} of {result['row_count']} rows):")
                    for idx, row in enumerate(result['results'][:MAX_PRINT_ROWS], 1):
                        print(f"   {idx}. {row}")
                    print(f"   💡 See CSV file for complete results")
            else:
                print(f"\n❌ Error: {result.get('error')}")
                print("💡 Try rephrasing your question or asking about table structure first.")
                
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            print(f"\n❌ Unexpected error: {e}")


if __name__ == "__main__":
    asyncio.run(main())