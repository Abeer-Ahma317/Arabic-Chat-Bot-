from typing import Dict, Any
import pandas as pd
import mysql.connector
import sqlite3
import re
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

class AIDataQuery:
    """
    AI-powered Data Query Agent
    Combines SQLCoder-7B with 7-layer text processing for Arabic queries
    """

    def __init__(self, db_config: Dict[str, Any] = None, db_type: str = "mysql",
                 db_path: str = None, auto_load_model: bool = True):
        """Initialize the AI DataQuery Agent"""
        print("\n" + "="*60)
        print("🤖 AI DataQuery Agent - AUTO INITIALIZATION")
        print("="*60)

        self.db_config = db_config
        self.db_type = db_type
        self.conn = None
        self.schema = ""
        self.db_entities = {}

        # Model components
        self.sqlcoder_model = None
        self.sqlcoder_tokenizer = None
        self.sqlcoder_pipeline = None

        # Initialize database
        print("\n📊 Step 1/2: Initializing Database...")
        db_success = self.initialize_database(db_path)

        if db_success:
            print("✅ Database connected and schema loaded!")

            if auto_load_model:
                print("\n🤖 Step 2/2: Loading AI Model...")
                print("⏳ This may take 2-5 minutes (one time only)...")

                model_success = self.load_sqlcoder_model()

                if model_success:
                    print("✅ SQLCoder Model loaded successfully!")
                else:
                    print("⚠️ Model loading failed - Agent will work in limited mode")
            else:
                print("\n⏭️ Step 2/2: Model loading skipped")

            print("\n" + "="*60)
            print("✅ Agent is FULLY OPERATIONAL!")
            print("="*60 + "\n")
        else:
            print("❌ Database initialization failed!")
            print("⚠️ Agent cannot function without database connection")
            print("="*60 + "\n")
    # =========================================================================
    # DATABASE METHODS
    # =========================================================================
          
    def get_mysql_connection(self):
        """Create MySQL connection"""
        try:
            conn = mysql.connector.connect(**self.db_config)
            print("✅ Connected to MySQL database")
            return conn
        except Exception as e:
            print(f"❌ Database error: {e}")
            return None

    def get_sqlite_connection(self, db_path: str):
        """Create SQLite connection"""
        try:
            conn = sqlite3.connect(db_path)
            print("✅ Connected to SQLite database")
            return conn
        except Exception as e:
            print(f"❌ Database error: {e}")
            return None

    def initialize_database(self, db_path: str = None):
        """Initialize database and load schema"""
        try:
            if self.db_type == "mysql":
                self.conn = self.get_mysql_connection()
                if not self.conn:
                    return False
                
                # Get all tables
                query_tables = "SHOW TABLES;"
                tables = pd.read_sql(query_tables, self.conn)
                self.schema = ""

                for table in tables.iloc[:, 0].tolist():
                    cols = pd.read_sql(f"DESCRIBE {table};", self.conn)
                    self.schema += f"\nCREATE TABLE {table} (\n"
                    col_defs = [f"    {row['Field']} {row['Type']}" 
                              for _, row in cols.iterrows()]
                    self.schema += ",\n".join(col_defs)
                    self.schema += "\n);\n"

            else:  # SQLite
                if not db_path:
                    print("❌ SQLite requires db_path")
                    return False

                self.conn = self.get_sqlite_connection(db_path)
                if not self.conn:
                    return False

                tables = pd.read_sql(
                    "SELECT name FROM sqlite_master WHERE type='table';", 
                    self.conn
                )
                self.schema = ""

                for table in tables['name']:
                    if table != 'sqlite_sequence':
                        cols = pd.read_sql(f"PRAGMA table_info([{table}]);", self.conn)
                        self.schema += f"\nCREATE TABLE {table} (\n"
                        col_defs = [f"    {col['name']} {col['type']}" 
                                  for _, col in cols.iterrows()]
                        self.schema += ",\n".join(col_defs)
                        self.schema += "\n);\n"

            print("✅ Schema loaded successfully")
            return True

        except Exception as e:
            print(f"❌ Database error: {e}")
            return False

    # =========================================================================
    # MODEL METHODS
    # =========================================================================

    def load_sqlcoder_model(self):
        """Load SQLCoder-7B model"""
        try:
            import torch
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            print(f"Using device: {device}")

            model_name = "defog/sqlcoder-7b-2"
            print(f"Loading SQLCoder from: {model_name}")

            # Load Tokenizer
            self.sqlcoder_tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                trust_remote_code=True
            )

            if self.sqlcoder_tokenizer.pad_token is None:
                self.sqlcoder_tokenizer.pad_token = self.sqlcoder_tokenizer.eos_token
                self.sqlcoder_tokenizer.pad_token_id = self.sqlcoder_tokenizer.eos_token_id

            # Load Model
            self.sqlcoder_model = AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map="auto",
                torch_dtype=torch.float16,
                trust_remote_code=True,
                load_in_8bit=True
            )
            print("SQLCoder Model loaded successfully")

            # Create Pipeline
            self.sqlcoder_pipeline = pipeline(
                "text-generation",
                model=self.sqlcoder_model,
                tokenizer=self.sqlcoder_tokenizer,
                max_new_tokens=300,
                temperature=0.1,
                do_sample=False,
                return_full_text=False,
                pad_token_id=self.sqlcoder_tokenizer.pad_token_id,
                eos_token_id=self.sqlcoder_tokenizer.eos_token_id
            )

            return True

        except Exception as e:
            print(f"❌ Model error: {e}")
            return False

    # =========================================================================
    # 7-LAYER TEXT PROCESSING
    # =========================================================================

    def layer1_text_normalization(self, question: str) -> Dict[str, Any]:
        """Layer 1: Arabic text normalization"""
        def normalize_arabic(text: str) -> str:
            if pd.isna(text):
                return ""
            text = str(text).strip()
            text = text.replace("أ", "ا").replace("إ", "ا").replace("آ", "ا")
            text = text.replace("ؤ", "و").replace("ئ", "ي").replace("ة", "ه")
            text = text.replace("ـ", "")
            return text

        normalized = normalize_arabic(question)
        tokens = [word.strip() for word in normalized.split() if word.strip()]

        return {
            'original': question,
            'normalized': normalized,
            'tokens': tokens,
            'clean_text': ' '.join(tokens)
        }

    def layer2_intent_classification(self, processed_text: Dict[str, Any]) -> Dict[str, Any]:
        """Layer 2: Intent classification"""
        text = processed_text['normalized'].lower()

        intent_patterns = {
            'COUNT_QUERY': ['كم', 'عدد', 'كام'],
            'LIST_QUERY': ['اسماء','ما', 'ما هي', 'اعرض', 'اظهر'],
            'AGGREGATE_QUERY': ['متوسط', 'اعلى','اكثر', 'اقل', 'مجموع'],
            'FILTER_QUERY': ['من','و', 'التي', 'الذين'],
            'JOIN_QUERY': ['معهد', 'موقع', 'منطقة', 'مركز', 'مهنة', 'اقليم']
        }

        detected_intent = 'GENERAL_QUERY'
        confidence = 0.0

        for intent, keywords in intent_patterns.items():
            matches = sum(1 for keyword in keywords if keyword in text)
            if matches > 0:
                current_confidence = matches / len(keywords)
                if current_confidence > confidence:
                    detected_intent = intent
                    confidence = current_confidence

        return {
            'intent': detected_intent,
            'confidence': confidence
        }

    def layer3_entity_recognition(self, processed_text: Dict, db_entities: dict = None) -> dict:
        """Layer 3: Entity extraction"""
        if isinstance(processed_text, str):
            processed_text = {"normalized": processed_text}

        text = processed_text['normalized']
        entities = []

        entity_patterns = {
            "phoneNumber": r"\b07\d{8}\b",
            "nationalID": r"\b\d{10}\b",         # الرقم الوطني - 10 أرقام
            "dateOfBirth": r"\b\d{4}-\d{2}-\d{2}\b",  # تاريخ الميلاد (yyyy-mm-dd)
            "mark": r"\b\d{1,3}\b",         # علامة امتحان (0-100 أو أكثر)
            'regionName': {
                'الشمالي': 'NORTHERN',
                'شمالي': 'NORTHERN',
                'الجنوبي': 'SOUTHERN',
                'جنوبي': 'SOUTHERN',
                'الأوسط': 'CENTRAL',
                'اوسط': 'CENTRAL',
                'أوسط': 'CENTRAL',
                'الشمال': 'NORTHERN',
                'الجنوب': 'SOUTHERN',
                'الوسط': 'CENTRAL'
            },
            'status': {
                'متقدم': 'PENDING',
                'ملتحق': 'PENDING',
                'قيد': 'PHONE_CALL',
                'اتصال': 'PHONE_CALL',
                'رفع الملفات': 'WAITING_FOR_DOCUMENTS',
                'قائمة انتظار': 'WAITING_FOR_DOCUMENTS',
                'ناجح': 'PASSED_THE_EXAM',
                'مقبول': 'ACCEPTED',
                'قبول': 'ACCEPTED',
                'مرفوض': 'REJECTED',
                'رفض': 'REJECTED'
            },
            'profession': {
                'ميكانيكي': 'ميكانيكي',
                'مشغل': 'مشغل',
                'حلاق': 'حلاق',
                'تمديدات': 'تمديدات',
                'خياط': 'خياط',
                'دهان': 'دهان' ,
                'مجهز': 'مجهز',
                'مركب': 'مركب',
                'لحيم': 'لحيم',
                'حداد': 'حداد',
                'نجار': 'نجار',
                'كهربائي': 'كهربائي',
                'داعم': 'داعم',
                'قصير': 'قصير' ,
                'بليط': 'بليط',
                'مزارع': 'مزارع',
                'التسويق': 'التسويق' ,
                'طاقة شمسية': 'طاقة شمسية'
            },
            'gender': {
                'ذكر': 'MALE',
                'ذكور': 'MALE',
                'انثى': 'FEMALE',
                'اناث': 'FEMALE',
                'أنثى': 'FEMALE'
            },
            'educationLevel': {
                'بكالوريوس': 'BACKELOR',
                'دبلوم': 'DIPLOMA',
                'ماجستير': 'MASTER',
                'ماستر': 'MASTER',
                'ثانوية عامة': 'HIGH_SCHOOL',
                'ثانوي': 'HIGH_SCHOOL',
                'ثانوية': 'HIGH_SCHOOL',
                'توجيهي': 'HIGH_SCHOOL',
                'اعدادي': 'MIDDLE_SCHOOL',
                'اعدادية': 'MIDDLE_SCHOOL',
                'متوسط': 'MIDDLE_SCHOOL'

            },
            'area': {
                'الزرقاء': 'الزرقاء',
                'الموقر': 'الموقر',
                'مادبا': 'مادبا',
                'ماركا الصناعي': 'ماركا الصناعي',
                'ذيبان ': 'ذيبان',
                'الكرك': 'الكرك',
                'معان': 'معان' ,
                'الطفيلة': 'الطفيلة',
                'القويرة': 'القويرة',
                'الريشة': 'الريشة ',
                'السرحان': 'السرحان',
                'عجلون': 'عجلون',
                'الرمثا': 'الرمثا',
                'جرش': 'جرش',
                'الجفر': 'الجفر' ,
                'المفرق': 'المفرق',
                'الكورة': 'الكورة'
            },
            'institute': {
                'معهد الزرقاء': 'معهد الزرقاء',
                'معهد الموقر': 'معهد الموقر',
                'معهد مادبا': 'معهد مادبا',
                'معهد ماركا الصناعي': 'معهد ماركا الصناعي',
                'معهد ذيبان ': 'معهد ذيبان',
                'معهد الكرك': 'معهد الكرك',
                'معهد معان': 'معهد معان' ,
                'معهد الطفيلة': 'معهد الطفيلة',
                'معهد القويرة': 'معهد القويرة',
                'معهد الريشة': 'معهد الريشة ',
                'معهد السرحان': 'معهدالسرحان',
                'معهد عجلون': 'معهد عجلون',
                'معهد مادبا': 'معهد الرمثا',
                'معهد جرش': 'معهد جرش',
                'معهد الطاقة الشمسية': 'مركز التميز الأردني الألماني للطاقة الشمسية' ,
                'معهد الجفر': 'معهد الجفر',
                'معهد الكورة': 'معهد الكورة'
            }
              }

        if db_entities:
            entity_patterns.update(db_entities)

        for entity_type, patterns in entity_patterns.items():
            if isinstance(patterns, str):  # regex
                match = re.search(patterns, text)
                if match:
                    entities.append({
                        'type': entity_type,
                        'value': match.group(),
                        'original': match.group()
                    })
            elif isinstance(patterns, dict):  # dictionary
                for pattern, standard_value in patterns.items():
                    if pattern in text:
                        entities.append({
                            'type': entity_type,
                            'value': standard_value,
                            'original': pattern
                        })

        return {
            'entities': entities,
            'entity_count': len(entities),
            'entity_types': list(set([e['type'] for e in entities]))
        }

    def layer4_context_analysis(self, processed_text: Dict[str, Any], 
                              intent: Dict[str, Any], 
                              entities: Dict[str, Any]) -> Dict[str, Any]:
        """Layer 4: Context analysis"""
        text = processed_text['normalized']

        table_indicators = {
            'userForm': ['طالب', 'طلاب','مسجل','متدرب', 'دارس'],
            'regions': ['اقليم', 'الاقليم', 'اقاليم'],
            'areas': ['منطقة', 'مكان السكن', 'سكان', 'موقع'],
            'institutes': ['معهد', 'معاهد', 'مراكز', 'مركز', 'مؤسسة'],
            'professions': ['مهنة', 'مهنته', 'المهنة']
        }

        required_tables = []
        for table, indicators in table_indicators.items():
            if any(indicator in text for indicator in indicators):
                required_tables.append(table)

        if not required_tables:
            required_tables = ['userForm']

        join_needed = len(required_tables) > 1

        return {
            'required_tables': required_tables,
            'join_needed': join_needed,
            'complexity_level': 'HIGH' if join_needed else 'LOW'
        }

    def layer5_schema_mapping(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Layer 5: Schema mapping"""
        schema_info = {}
        current_table = None

        for line in self.schema.split('\n'):
            line = line.strip()
            if line.startswith('CREATE TABLE'):
                current_table = line.split()[2].strip('(')
                schema_info[current_table] = []
            elif current_table and line and not line.startswith('CREATE'):
                schema_info[current_table].append(line)

        relevant_schema = {}
        for table in context['required_tables']:
            if table in schema_info:
                relevant_schema[table] = schema_info[table]

        return {
            'relevant_schema': relevant_schema,
            'full_schema': schema_info
        }

    def layer6_prompt_optimization(self, question: str, intent: Dict, 
                                 entities: Dict, context: Dict, 
                                 schema_mapping: Dict) -> str:
        """Layer 6: Prompt optimization"""
        prompt = f"""
### Task
Generate a SQL query to answer the following Arabic question: `{question}`

### Database Schema
{self.schema}

### Additional Context
- Question Type: {intent['intent']}
- Entities: {', '.join([e['value'] for e in entities['entities']])}
- Tables Needed: {', '.join(context['required_tables'])}
- For Arabic text matching, use LIKE '%text%'
- IMPORTANT: Do NOT use ILIKE. Use only LIKE. SQLite does not support ILIKE.
  Example: use "WHERE column LIKE '%text%'" instead of "ILIKE".

### Relationships:
    - userForm.region → regions.name
    - userForm.area → areas.name
    - userForm.institute → institutes.name
    - areas.regionName → regions.name
    - institutes.areaName → areas.name
    - institutes.regionName → regions.name
    - professions.areaName → areas.name
    - professions.regionName → regions.name


### Answer
Given the database schema, here is the SQL query that answers `{question}`:
```sql
"""
        return prompt

    def layer7_response_processing(self, raw_response: str) -> str:
        """Layer 7: Process SQLCoder response"""
        if not raw_response:
            return None
        # Extract from code block
        sql_block_match = re.search(r'```sql\n(.*?)```', raw_response, re.DOTALL)
        if sql_block_match:
            sql = sql_block_match.group(1).strip()
            return sql.replace(';', '').strip()
        # Extract SELECT statement
        lines = raw_response.split('\n')
        for line in lines:
            line = line.strip()
            if line.upper().startswith('SELECT'):
                return line.replace(';', '').strip()
        # Regex fallback
        sql_match = re.search(r'SELECT[^;]*', raw_response, re.IGNORECASE | re.DOTALL)
        if sql_match:
            return sql_match.group(0).strip().replace(';', '')

        return None


    # =========================================================================
    # MAIN PROCESSING METHODS
    # =========================================================================

    def send_to_sqlcoder(self, prompt: str) -> str:
        """Send prompt to SQLCoder"""
        if not self.sqlcoder_pipeline:
            print("SQLCoder not available!")
            return None

        try:
            response = self.sqlcoder_pipeline(prompt, max_new_tokens=300)
            if isinstance(response, list) and len(response) > 0:
                return response[0].get('generated_text', '')
            return str(response)
        except Exception as e:
            print(f"Error calling SQLCoder: {e}")
            return None

    def process_sql_question(self, question: str) -> Dict[str, Any]:
        """Process question through all layers"""
        print(f"Processing question: {question}")
        print("=" * 60)

        layer1_result = self.layer1_text_normalization(question)
        print(f"Layer 1: Normalized text")

        layer2_result = self.layer2_intent_classification(layer1_result)
        print(f"Layer 2: Intent: {layer2_result['intent']}")

        layer3_result = self.layer3_entity_recognition(layer1_result)
        print(f"Layer 3: Found {layer3_result['entity_count']} entities")

        layer4_result = self.layer4_context_analysis(
            layer1_result, layer2_result, layer3_result
        )
        print(f"Layer 4: Required tables: {layer4_result['required_tables']}")

        layer5_result = self.layer5_schema_mapping(layer4_result)
        print(f"Layer 5: Schema mapped: {len(layer5_result['relevant_schema'])} tables")

        optimized_prompt = self.layer6_prompt_optimization(
            question, layer2_result, layer3_result, 
            layer4_result, layer5_result
        )
        print(f"Layer 6: Prompt created ({len(optimized_prompt)} chars)")

        # Send to SQLCoder
        print("Sending to SQLCoder...")
        raw_response = self.send_to_sqlcoder(optimized_prompt)
        
        # Layer 7
        final_sql = self.layer7_response_processing(raw_response)
        print(f"Layer 7: SQL generated")

        return {
            'question': question,
            'final_sql': final_sql,
            'success': final_sql is not None,
            'raw_response': raw_response
        }

    def classify_intent(self, user_text: str) -> str:
        """Classify user intent"""
        lower_text = user_text.lower()

        chat_keywords = ['مرحبا', 'اهلا', 'السلام', 'صباح', 'مساء', 
                        'كيف حالك', 'شو اخبارك', 'هاي', 'هلا', 'شكرا', 
                        'باي', 'وداعا', 'من انت', 'شو انت', 'كيفك']

        if any(kw in lower_text for kw in chat_keywords):
            return "chat"

        sql_keywords = ['كم', 'عدد', 'اسماء', 'اعرض', 'اظهر', 'من', 
                       'ما', 'طالب', 'طلاب', 'علامة', 'معهد', 'منطقة']

        if any(kw in lower_text for kw in sql_keywords):
            return "sql"

        return "general"

    def chat_response(self, user_text: str) -> str:
        """Generate chat response"""
        lower_text = user_text.lower()

        if any(w in lower_text for w in ['مرحبا', 'اهلا', 'السلام']):
            return """مرحباً! 👋 
أنا بوت ذكي للاستعلام عن قاعدة البيانات.

🎯 جرب الأسئلة التالية:
• كم عدد الطلاب؟
• من الطلاب في الشمال؟
• كم عدد الإناث؟"""

        if any(w in lower_text for w in ['كيف حالك', 'شو اخبارك']):
            return "الحمد لله! 😊 كيف أقدر أساعدك؟"

        if any(w in lower_text for w in ['من انت', 'شو انت']):
            return """أنا بوت ذكي 🤖 
أساعد في الاستعلام عن قاعدة البيانات بالعربي!"""

        if any(w in lower_text for w in ['شكرا', 'مشكور']):
            return "العفو! 🌟"

        return """مرحباً! 👋 كيف أقدر أساعدك؟

💡 جرب:
• كم عدد الطلاب؟
• من الطلاب في الشمال؟
• كم عدد الإناث؟"""

    def execute_query(self, question: str) -> Dict[str, Any]:
        """Execute query with enhanced processing"""
        intent = self.classify_intent(question)
        print(f"\n🎯 Intent: {intent}")

        if intent == "chat":
            return {
                'success': True,
                'intent': 'chat',
                'answer': self.chat_response(question),
                'sql': None,
                'results': None
            }

        if intent == "general":
            return {
                'success': False,
                'intent': 'general',
                'answer': "عذراً، لم أفهم السؤال. جرب سؤالاً آخر!",
                'sql': None,
                'results': None
            }

        result = self.process_sql_question(question)
        if not result['success']:
            return {
                'success': False,
                'intent': 'sql',
                'answer': '⚠️ عذراً، لم أفهم السؤال',
                'sql': None,
                'results': None,
                'debug': result.get('raw_response')
            }
              # Execute SQL
        max_attempts = 2
        last_error = None
        for attempt in range(max_attempts):
            try:
                df = pd.read_sql(result['final_sql'], self.conn)
                results_list = df.to_dict('records')

                if len(results_list) == 1 and len(results_list[0]) == 1:
                    value = list(results_list[0].values())[0]
                    answer = f"✅ النتيجة: {value}"
                elif len(results_list) > 0:
                    answer = f"✅ تم العثور على {len(results_list)} نتيجة"
                else:
                    answer = "ℹ️ لا توجد نتائج"

                return {
                    'success': True,
                    'intent': 'sql',
                    'answer': answer,
                    'sql': result['final_sql'],
                    'results': results_list[:]
                }
            except Exception as e:
                last_error = str(e)
                print(f"❌ Attempt {attempt + 1} failed: {last_error}")

                if attempt < max_attempts - 1:
                    sql_query = result['final_sql']
                    if "ILIKE" in sql_query:
                        sql_query = sql_query.replace("ILIKE", "LIKE COLLATE NOCASE")
                    if "ilike" in sql_query:
                        sql_query = sql_query.replace("ilike", "LIKE COLLATE NOCASE")

                    if not result['final_sql'].endswith(';'):
                        result['final_sql'] += ';'
        return {
            'success': False,
            'intent': 'sql',
            'answer': "عذراً، لم أفهم السؤال. جرب سؤالاً آخر!",
            'sql': None,
            'results': None
        }
                    

    def query(self, question: str) -> Dict[str, Any]:
        """Main query method"""
        return self.execute_query(question)

    def close(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()
            print("✅ Database connection closed")