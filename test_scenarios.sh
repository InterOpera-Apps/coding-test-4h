#!/bin/bash

# Test Scenarios Verification Script
# Based on README.md lines 493-517

DOCUMENT_ID=164
BASE_URL="http://localhost:8000/api"

echo "=========================================="
echo "Testing Scenarios from README.md"
echo "=========================================="
echo ""

# Scenario 2: Text-based Question
echo "📝 Scenario 2: Text-based Question"
echo "Question: 'What is the main contribution of this paper?'"
echo "Expected: Should mention Transformer architecture and attention mechanisms"
echo ""

RESPONSE=$(curl -s -X POST "$BASE_URL/chat" \
  -H "Content-Type: application/json" \
  -d "{\"message\": \"What is the main contribution of this paper?\", \"document_id\": $DOCUMENT_ID}")

echo "$RESPONSE" | python3 -c "
import sys, json
try:
    data = json.load(sys.stdin)
    answer = data.get('answer', 'No answer')
    print(f\"Answer: {answer[:200] if answer else 'No answer'}...\")
    sources = data.get('sources', []) or []
    print(f\"Sources: {len(sources)} sources found\")
    print(f\"Processing time: {data.get('processing_time', 0)}s\")
    print(f\"Conversation ID: {data.get('conversation_id', 'N/A')}\")
except Exception as e:
    print(f\"Error: {e}\", file=sys.stderr)
    sys.exit(1)
"
CONV_ID=$(echo "$RESPONSE" | python3 -c "import sys, json; data = json.load(sys.stdin); print(data.get('conversation_id', ''))" 2>/dev/null || echo "")

echo ""
echo "---"
echo ""

# Scenario 3: Image-related Question
echo "🖼️  Scenario 3: Image-related Question"
echo "Question: 'Show me the Transformer architecture diagram'"
echo "Expected: Should display Figure 1 (Transformer model architecture)"
echo ""

RESPONSE=$(curl -s -X POST "$BASE_URL/chat" \
  -H "Content-Type: application/json" \
  -d "{\"message\": \"Show me the Transformer architecture diagram\", \"document_id\": $DOCUMENT_ID, \"conversation_id\": $CONV_ID}")

echo "$RESPONSE" | python3 -c "
import sys, json
try:
    data = json.load(sys.stdin)
    answer = data.get('answer', 'No answer')
    print(f\"Answer: {answer[:200] if answer else 'No answer'}...\")
    sources = data.get('sources', []) or []
    images = [s for s in sources if s and isinstance(s, dict) and s.get('type') == 'image']
    print(f\"Images found: {len(images)}\")
    for i, img in enumerate(images[:3], 1):
        if img and isinstance(img, dict):
            caption = img.get('caption')
            caption_str = str(caption)[:80] if caption else 'No caption'
            print(f\"  {i}. {caption_str}...\")
    print(f\"Total sources: {len(sources)}\")
except Exception as e:
    print(f\"Error: {e}\", file=sys.stderr)
    import traceback
    traceback.print_exc(file=sys.stderr)
"

echo ""
echo "---"
echo ""

# Scenario 4: Table-related Question
echo "📊 Scenario 4: Table-related Question"
echo "Question: 'What are the BLEU scores for different models?'"
echo "Expected: Should display performance comparison tables"
echo ""

RESPONSE=$(curl -s -X POST "$BASE_URL/chat" \
  -H "Content-Type: application/json" \
  -d "{\"message\": \"What are the BLEU scores for different models?\", \"document_id\": $DOCUMENT_ID, \"conversation_id\": $CONV_ID}")

echo "$RESPONSE" | python3 -c "
import sys, json
try:
    data = json.load(sys.stdin)
    answer = data.get('answer', 'No answer')
    print(f\"Answer: {answer[:200] if answer else 'No answer'}...\")
    sources = data.get('sources', []) or []
    tables = [s for s in sources if s and isinstance(s, dict) and s.get('type') == 'table']
    print(f\"Tables found: {len(tables)}\")
    for i, tbl in enumerate(tables[:3], 1):
        if tbl and isinstance(tbl, dict):
            page = tbl.get('page') if tbl.get('page') is not None else '?'
            caption = tbl.get('caption')
            caption_str = str(caption)[:60] if caption else 'No caption'
            print(f\"  {i}. Page {page}: {caption_str}...\")
    print(f\"Total sources: {len(sources)}\")
except Exception as e:
    print(f\"Error: {e}\", file=sys.stderr)
    import traceback
    traceback.print_exc(file=sys.stderr)
"

echo ""
echo "---"
echo ""

# Scenario 5: Multi-turn Conversation
echo "💬 Scenario 5: Multi-turn Conversation"
echo "First question: 'What attention mechanism does this paper propose?'"
echo ""

RESPONSE1=$(curl -s -X POST "$BASE_URL/chat" \
  -H "Content-Type: application/json" \
  -d "{\"message\": \"What attention mechanism does this paper propose?\", \"document_id\": $DOCUMENT_ID, \"conversation_id\": $CONV_ID}")

echo "$RESPONSE1" | python3 -c "
import sys, json
try:
    data = json.load(sys.stdin)
    answer = data.get('answer', 'No answer')
    print(f\"Answer 1: {answer[:200] if answer else 'No answer'}...\")
except Exception as e:
    print(f\"Error: {e}\", file=sys.stderr)
"

echo ""
echo "Follow-up: 'How does it compare to RNN and CNN?'"
echo "Expected: Second answer should reference the first question's context"
echo ""

RESPONSE2=$(curl -s -X POST "$BASE_URL/chat" \
  -H "Content-Type: application/json" \
  -d "{\"message\": \"How does it compare to RNN and CNN?\", \"document_id\": $DOCUMENT_ID, \"conversation_id\": $CONV_ID}")

echo "$RESPONSE2" | python3 -c "
import sys, json
try:
    data = json.load(sys.stdin)
    answer = data.get('answer', '')
    if answer:
        print(f\"Answer 2: {answer[:300]}...\")
        # Check if answer references previous context
        if 'attention' in answer.lower() or 'transformer' in answer.lower():
            print(\"✅ Answer references previous context\")
        else:
            print(\"⚠️  Answer may not reference previous context\")
    else:
        print(\"No answer received\")
except Exception as e:
    print(f\"Error: {e}\", file=sys.stderr)
"

echo ""
echo "---"
echo ""

# Scenario 6: Specific Technical Question
echo "🔬 Scenario 6: Specific Technical Question"
echo "Question: 'What is the computational complexity of self-attention?'"
echo "Expected: Should mention O(n²·d) complexity and reference Section 3.2.1"
echo ""

RESPONSE=$(curl -s -X POST "$BASE_URL/chat" \
  -H "Content-Type: application/json" \
  -d "{\"message\": \"What is the computational complexity of self-attention?\", \"document_id\": $DOCUMENT_ID, \"conversation_id\": $CONV_ID}")

echo "$RESPONSE" | python3 -c "
import sys, json
try:
    data = json.load(sys.stdin)
    answer = data.get('answer', '')
    if answer:
        print(f\"Answer: {answer[:400]}...\")
        # Check for expected content
        has_complexity = 'O(n' in answer or 'O(n²' in answer or 'quadratic' in answer.lower()
        has_section = '3.2' in answer or 'section' in answer.lower()
        print(f\"\\n✅ Contains complexity notation: {has_complexity}\")
        print(f\"✅ References section: {has_section}\")
    else:
        print(\"No answer received\")
    sources = data.get('sources', []) or []
    print(f\"Sources: {len(sources)} found\")
except Exception as e:
    print(f\"Error: {e}\", file=sys.stderr)
    import traceback
    traceback.print_exc(file=sys.stderr)
"

echo ""
echo "=========================================="
echo "All scenarios tested!"
echo "=========================================="

