"""
Few-shot Question -> SQL examples for TTMD Bot.

HOW THIS FILE IS USED
- ttmd_bot.py loads EXAMPLES, checks every query against your live database
  (parser check + MySQL EXPLAIN), and only stores the ones that pass.
- Examples that fail are skipped with a warning, so a wrong column name never
  reaches the AI.
- Edit or add examples freely. On the next start the vector store updates itself.

RULES FOR NEW EXAMPLES
- Use only real table / column names and real values from your database.
- One read-only query per example (SELECT or WITH ... SELECT), no semicolon.
- Do NOT use curly braces { } in a query (LangChain treats them as placeholders).

WHAT THESE EXAMPLES ARE BASED ON (confirmed from your own query results)
- Table:   modeldata
- Columns: CustomerName, DeviceTypeName, Recency_Category
- Values:  Recency_Category = '1. Active (< 24 hrs)' and '4. Stale (> 30 Days)'
           DeviceTypeName includes 'SIMHA - 01'

NOT COVERED YET (column names not confirmed, so nothing was invented)
- Online / Offline status, operator (for example AirTel), date filters,
  MAX / MIN, SUM / AVG on numeric columns.
  Run `schema` in the bot, send me the column list, and I will add them.

NOTE ON CustomerName
- Earlier results showed one device per CustomerName value, so it may be a
  site or device code rather than a real customer. The two customer examples
  below are valid SQL, but check them against what CustomerName really means.
"""

EXAMPLES = [
    # ---------- Counting records ----------
    {
        "category": "count",
        "question": "How many devices are there in total?",
        "query": "SELECT COUNT(*) AS total_devices FROM modeldata",
    },
    {
        "category": "count",
        "question": "How many different device types are there?",
        "query": "SELECT COUNT(DISTINCT DeviceTypeName) AS device_type_count FROM modeldata",
    },
    # ---------- Filtering by status ----------
    {
        "category": "filter",
        "question": "How many devices were active in the last 24 hours?",
        "query": (
            "SELECT COUNT(*) AS active_devices FROM modeldata "
            "WHERE Recency_Category = '1. Active (< 24 hrs)'"
        ),
    },
    {
        "category": "filter",
        "question": "How many devices are stale for more than 30 days?",
        "query": (
            "SELECT COUNT(*) AS stale_devices FROM modeldata "
            "WHERE Recency_Category = '4. Stale (> 30 Days)'"
        ),
    },
    # ---------- Multiple conditions ----------
    {
        "category": "multi_condition",
        "question": "How many stale devices (more than 30 days) are of device type SIMHA - 01?",
        "query": (
            "SELECT COUNT(*) AS stale_devices FROM modeldata "
            "WHERE Recency_Category = '4. Stale (> 30 Days)' "
            "AND DeviceTypeName = 'SIMHA - 01'"
        ),
    },
    # ---------- Grouping ----------
    {
        "category": "group_by",
        "question": "How many devices are there for each device type?",
        "query": (
            "SELECT DeviceTypeName, COUNT(*) AS device_count FROM modeldata "
            "GROUP BY DeviceTypeName ORDER BY device_count DESC"
        ),
    },
    {
        "category": "group_by",
        "question": "How many devices are there in each recency category?",
        "query": (
            "SELECT Recency_Category, COUNT(*) AS device_count FROM modeldata "
            "GROUP BY Recency_Category ORDER BY Recency_Category"
        ),
    },
    {
        "category": "group_by_customer",
        "question": "How many devices does each customer have?",
        "query": (
            "SELECT CustomerName, COUNT(*) AS device_count FROM modeldata "
            "GROUP BY CustomerName ORDER BY device_count DESC, CustomerName LIMIT 20"
        ),
    },
    # ---------- Sorting and ranking ----------
    {
        "category": "ranking",
        "question": "Which 5 device types have the most stale devices?",
        "query": (
            "SELECT DeviceTypeName, COUNT(*) AS stale_devices FROM modeldata "
            "WHERE Recency_Category = '4. Stale (> 30 Days)' "
            "GROUP BY DeviceTypeName ORDER BY stale_devices DESC LIMIT 5"
        ),
    },
    {
        "category": "ranking",
        "question": "Which 10 customers have the most stale devices?",
        "query": (
            "SELECT CustomerName, COUNT(*) AS stale_devices FROM modeldata "
            "WHERE Recency_Category = '4. Stale (> 30 Days)' "
            "GROUP BY CustomerName ORDER BY stale_devices DESC, CustomerName LIMIT 10"
        ),
    },
    # ---------- Percentages ----------
    {
        "category": "percentage",
        "question": "What percentage of all devices are stale for more than 30 days?",
        "query": (
            "SELECT ROUND(100 * SUM(Recency_Category = '4. Stale (> 30 Days)') / COUNT(*), 2) "
            "AS stale_percentage FROM modeldata"
        ),
    },
    {
        "category": "percentage",
        "question": "What is the share of devices in each recency category as a percentage?",
        "query": (
            "SELECT Recency_Category, COUNT(*) AS device_count, "
            "ROUND(100 * COUNT(*) / (SELECT COUNT(*) FROM modeldata), 2) AS percentage "
            "FROM modeldata GROUP BY Recency_Category ORDER BY Recency_Category"
        ),
    },
    # ---------- Comparing categories ----------
    {
        "category": "comparison",
        "question": "Compare the number of active and stale devices for each device type.",
        "query": (
            "SELECT DeviceTypeName, "
            "SUM(Recency_Category = '1. Active (< 24 hrs)') AS active_devices, "
            "SUM(Recency_Category = '4. Stale (> 30 Days)') AS stale_devices "
            "FROM modeldata GROUP BY DeviceTypeName ORDER BY stale_devices DESC LIMIT 10"
        ),
    },
    # ---------- CTE ----------
    {
        "category": "cte",
        "question": "Which device types have the highest percentage of stale devices?",
        "query": (
            "WITH type_stats AS ("
            "SELECT DeviceTypeName, COUNT(*) AS total_devices, "
            "SUM(Recency_Category = '4. Stale (> 30 Days)') AS stale_devices "
            "FROM modeldata GROUP BY DeviceTypeName) "
            "SELECT DeviceTypeName, total_devices, stale_devices, "
            "ROUND(100 * stale_devices / total_devices, 2) AS stale_percentage "
            "FROM type_stats ORDER BY stale_percentage DESC, total_devices DESC LIMIT 5"
        ),
    },
    # ---------- More direct examples (active %, side-by-side totals, one device type) ----------
    {
        "category": "percentage",
        "question": "What percentage of devices were active in the last 24 hours?",
        "query": (
            "SELECT ROUND(100 * SUM(Recency_Category = '1. Active (< 24 hrs)') / COUNT(*), 2) "
            "AS active_percentage FROM modeldata"
        ),
    },
    {
        "category": "comparison",
        "question": "How many devices are active and how many are stale?",
        "query": (
            "SELECT SUM(Recency_Category = '1. Active (< 24 hrs)') AS active_devices, "
            "SUM(Recency_Category = '4. Stale (> 30 Days)') AS stale_devices FROM modeldata"
        ),
    },
    {
        "category": "ranking",
        "question": "Which device type has the most devices?",
        "query": (
            "SELECT DeviceTypeName, COUNT(*) AS device_count FROM modeldata "
            "GROUP BY DeviceTypeName ORDER BY device_count DESC LIMIT 1"
        ),
    },
    {
        "category": "percentage",
        "question": "What percentage of SIMHA - 01 devices are stale?",
        "query": (
            "SELECT ROUND(100 * SUM(Recency_Category = '4. Stale (> 30 Days)') / COUNT(*), 2) "
            "AS stale_percentage FROM modeldata WHERE DeviceTypeName = 'SIMHA - 01'"
        ),
    },
    {
        "category": "filter",
        "question": "How many unreliable devices are there?",
        "query": (
            "SELECT COUNT(*) AS unreliable_devices FROM modeldata "
            "WHERE Recency_Category = '4. Stale (> 30 Days)'"
        ),
    },
    # ---------- Subquery ----------
    {
        "category": "subquery",
        "question": "Which device types have more stale devices than the average device type?",
        "query": (
            "SELECT DeviceTypeName, COUNT(*) AS stale_devices FROM modeldata "
            "WHERE Recency_Category = '4. Stale (> 30 Days)' "
            "GROUP BY DeviceTypeName "
            "HAVING COUNT(*) > ("
            "SELECT AVG(stale_count) FROM ("
            "SELECT COUNT(*) AS stale_count FROM modeldata "
            "WHERE Recency_Category = '4. Stale (> 30 Days)' "
            "GROUP BY DeviceTypeName) AS per_type) "
            "ORDER BY stale_devices DESC"
        ),
    },
]