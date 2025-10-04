"""
Response handling 
"""
import pandas as pd
from typing import Dict, Any
from .models import classify_intent, chat_response
from .nlp_pipeline import extract_entities, determine_intent, analyze_context
from .processing import generate_sql, build_fallback_sql
from .database import get_connection

def execute_query(question: str) -> Dict[str, Any]:
    """Main implementation of the query""
    
    # Layer 1: 
    intent_type = classify_intent(question)
    
    # Normal conversation
    if intent_type == "chat":
        return {
            'success': True,
            'intent': 'chat',
            'answer': chat_response(question),
            'sql': None,
            'results': None
        }
    
    #Out of scope question
    if intent_type == "general":
        return {
            'success': False,
            'intent': 'general',
            'answer': 'عذراً، أنا متخصص فقط بقاعدة بيانات الطلاب.',
            'sql': None,
            'results': None
        }
    
    
    # Layer 3:
    entities = extract_entities(question)
    
    # Layer 2:
    query_intent = determine_intent(question, entities)
    
    # Layer 4: 
    context = analyze_context(question, entities, query_intent)
    
    # Layer 5 & 6 & 7: generate SQL
    result = generate_sql(question, entities, query_intent, context)
    final_sql = result.get('sql')
    
    # Fallback( If  generate_sql fails)
    if not final_sql:
        print("Using fallback SQL...")
        final_sql = build_fallback_sql(question, entities, query_intent)
    
    if not final_sql:
        return {
            'success': False,
            'intent': 'sql',
            'answer': 'عذراً، لم أتمكن من فهم السؤال.',
            'sql': None,
            'results': None
        }
    
    # تنفيذ SQL
    try:
        conn = get_connection()
        df = pd.read_sql(final_sql, conn)
        results = df.to_dict('records')
        
        # Answer format
        if len(results) == 1 and len(results[0]) == 1:
            value = list(results[0].values())[0]
            answer = f"النتيجة: {value}"
        elif len(results) > 0:
            answer = f"تم العثور على {len(results)} نتيجة"
        else:
            answer = "لا توجد نتائج"
        
        print(f"Success: {answer}\n")
        
        return {
            'success': True,
            'intent': 'sql',
            'answer': answer,
            'sql': final_sql,
            'results': results[:100]
        }
    
    except Exception as e:
        print(f"Execution error: {e}\n")
        return {
            'success': False,
            'intent': 'sql',
            'answer': f'خطأ في التنفيذ: {str(e)}',
            'sql': final_sql,
            'results': None
        }
