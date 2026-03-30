#!/bin/bash
# Memory System Health Check
# Run this to verify all upgrades are working

echo "════════════════════════════════════════════"
echo "  Memory System Health Check"
echo "════════════════════════════════════════════"
echo ""

# Color codes
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check 1: Embedding server
echo -n "1. Embedding server (port 8004)... "
if curl -s http://localhost:8004/health > /dev/null 2>&1; then
    echo -e "${GREEN}✅ Online${NC}"
else
    echo -e "${RED}❌ Offline${NC}"
fi

# Check 2: Qdrant
echo -n "2. Qdrant server (port 6333)... "
if curl -s http://localhost:6333/collections > /dev/null 2>&1; then
    echo -e "${GREEN}✅ Online${NC}"
    COLLECTIONS=$(curl -s http://localhost:6333/collections | python3 -c "import sys,json; print(len(json.load(sys.stdin)['result']['collections']))")
    echo "   └─ Collections: $COLLECTIONS"
else
    echo -e "${RED}❌ Offline${NC}"
fi

# Check 3: Anthropic API key
echo -n "3. Anthropic API key... "
if grep -q "ANTHROPIC_API_KEY=sk-" .env 2>/dev/null; then
    echo -e "${GREEN}✅ Configured${NC}"
else
    echo -e "${YELLOW}⚠️  Not found in .env${NC}"
fi

# Check 4: PM2 autocommit service
echo -n "4. Memory autocommit service... "
if pm2 list | grep -q "memory-autocommit.*online"; then
    echo -e "${GREEN}✅ Running${NC}"
    COMMITS=$(pm2 logs memory-autocommit --nostream --lines 5 | grep -c "Committed")
    if [ "$COMMITS" -gt 0 ]; then
        echo "   └─ Recent commits: $COMMITS"
    fi
else
    echo -e "${RED}❌ Not running${NC}"
    echo "   └─ Start with: pm2 start tools/memory_autocommit.py --name memory-autocommit --interpreter python3 -- --daemon"
fi

# Check 5: Cron job
echo -n "5. Weekly consolidation cron... "
if crontab -l 2>/dev/null | grep -q "consolidator.py memory"; then
    echo -e "${GREEN}✅ Configured${NC}"
    CRON_LINE=$(crontab -l | grep consolidator.py memory | head -1)
    echo "   └─ Schedule: Sunday 4 AM MSK"
else
    echo -e "${RED}❌ Not configured${NC}"
fi

# Check 6: Tools exist
echo ""
echo "6. Tool files:"
TOOLS=("proactive_memory.py" "memory_search_all.py" "consolidator.py memory" "memory_autocommit.py" "memory_apply_consolidation.py")
for tool in "${TOOLS[@]}"; do
    if [ -f "tools/$tool" ]; then
        echo -e "   ${GREEN}✅${NC} $tool"
    else
        echo -e "   ${RED}❌${NC} $tool (missing)"
    fi
done

# Check 7: State directory
echo ""
echo -n "7. State directory... "
if [ -d "tools/.memory_state" ]; then
    echo -e "${GREEN}✅ Exists${NC}"
else
    echo -e "${YELLOW}⚠️  Missing (will be auto-created)${NC}"
fi

# Quick functionality tests
echo ""
echo "════════════════════════════════════════════"
echo "  Quick Functionality Tests"
echo "════════════════════════════════════════════"
echo ""

# Test 1: Proactive memory
echo "Test 1: Proactive memory search"
python3 tools/proactive_memory.py "test query" 2>&1 | head -5
echo ""

# Test 2: Unified search
echo "Test 2: Unified search (first 3 results)"
python3 tools/memory_search_all.py "test" 2>&1 | head -15
echo ""

echo "════════════════════════════════════════════"
echo "  Health Check Complete"
echo "════════════════════════════════════════════"
echo ""
echo "📚 Full documentation: tools/MEMORY_SYSTEM_UPGRADE_COMPLETE.md"
echo "📖 Tool reference: tools/README.md"
echo ""
