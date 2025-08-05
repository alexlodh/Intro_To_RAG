#!/usr/bin/env python3
"""
Quick test to verify conversation memory is working
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from langchain_rag import langchain_rag_03_rag_pt2 as rag_module

def test_conversation_memory():
    """Test that conversation memory persists across multiple interactions."""
    print("Testing conversation memory...")
    
    # Setup (same as main function)
    vector_store = rag_module.setup_vector_store()
    self_query_retriever = rag_module.create_self_query_retriever(vector_store)
    
    # Create RAG chain
    rag_chain = rag_module.create_tool_based_conversational_rag_chain()
    config = {"configurable": {"thread_id": "memory_test"}}
    
    # First question
    print("\n=== First Question ===")
    question1 = "What AI developments happened in 2025?"
    print(f"Q1: {question1}")
    
    result1 = rag_chain.invoke({"question": question1}, config=config)
    answer1 = result1.get('answer', 'No answer')
    print(f"A1: {answer1[:200]}...")
    
    # Follow-up question that should reference previous context
    print("\n=== Follow-up Question (testing memory) ===")
    question2 = "Can you tell me more about the first one you mentioned?"
    print(f"Q2: {question2}")
    
    result2 = rag_chain.invoke({"question": question2}, config=config)
    answer2 = result2.get('answer', 'No answer')
    print(f"A2: {answer2[:200]}...")
    
    # Check if the answer references previous context
    if "G2" in answer2 or "AI" in answer2 or "categories" in answer2:
        print("✅ Memory test PASSED - Follow-up question understood previous context!")
    else:
        print("❌ Memory test FAILED - Follow-up question didn't reference previous context")
    
    return result1, result2

if __name__ == "__main__":
    test_conversation_memory()
