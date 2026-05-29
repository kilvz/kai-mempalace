"""Entity detection for Kai MemPalace — regex + heuristic named entity recognition.

Detects named entities (people, projects, technologies, organizations,
locations, dates, concepts, events) using pure pattern matching with
no external ML models or dependencies beyond stdlib.
"""

import functools
import json
import os
import re
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any

from kai_mempalace.i18n import get_entity_patterns

logger = logging.getLogger(__name__)

# ==================== COCA CONTENT-WORD FILTER ====================
# These are common English content words that frequently appear capitalized
# (sentence start, headings, markdown emphasis) but are NOT proper nouns.
# Filtering these at candidate-extraction time prevents false-positive entity
# detection of words like "Code", "Brutal", "Phase", etc.

_DATA_DIR = Path(__file__).parent / "data"


@functools.lru_cache(maxsize=1)
def _get_coca_filter() -> frozenset[str]:
    data_path = _DATA_DIR / "coca_content_words.json"
    try:
        raw = json.loads(data_path.read_text(encoding="utf-8"))
        words = raw.get("words", [])
        return frozenset(w.lower() for w in words if isinstance(w, str))
    except (OSError, json.JSONDecodeError, AttributeError, TypeError):
        return frozenset()


@functools.lru_cache(maxsize=1)
def _get_known_systems() -> tuple[tuple[str, "re.Pattern[str]"], ...]:
    data_path = _DATA_DIR / "known_systems.json"
    try:
        raw = json.loads(data_path.read_text(encoding="utf-8"))
        compounds = raw.get("compounds", [])
        valid = [c for c in compounds if isinstance(c, str) and c.strip()]
        sorted_compounds = sorted(valid, key=len, reverse=True)
        compiled: list[tuple[str, re.Pattern[str]]] = []
        for c in sorted_compounds:
            pattern = r"(?<!\w)" + re.escape(c) + r"(?!\w)"
            try:
                compiled.append((c, re.compile(pattern, re.IGNORECASE)))
            except re.error:
                continue
        return tuple(compiled)
    except (OSError, json.JSONDecodeError, AttributeError, TypeError):
        return ()


def _apply_known_systems_prepass(text: str) -> tuple[str, dict[str, int]]:
    compounds = _get_known_systems()
    counts: dict[str, int] = {}
    if not compounds:
        return text, counts
    working = list(text)
    for name, pattern in compounds:
        for m in pattern.finditer(text):
            counts[name] = counts.get(name, 0) + 1
            for i in range(m.start(), m.end()):
                working[i] = " "
    return "".join(working), counts


# File-type constants used by llm_refine and other downstream modules
PROSE_EXTENSIONS = {
    ".txt",
    ".md",
    ".rst",
    ".csv",
}

SKIP_DIRS = {
    ".git",
    "node_modules",
    "__pycache__",
    ".venv",
    "venv",
    "env",
    "dist",
    "build",
    ".next",
    "coverage",
    ".mempalace",
}

# ---------------------------------------------------------------------------
# Built-in knowledge base — names, technologies, locations
# ---------------------------------------------------------------------------

_KNOWN_TECHNOLOGIES: set[str] = {
    # Languages
    "Python", "JavaScript", "TypeScript", "Rust", "Go", "Golang", "Java",
    "Kotlin", "Swift", "Ruby", "PHP", "C++", "C#", "Scala", "Perl", "Lua",
    "Haskell", "Elixir", "Clojure", "Dart", "R", "Julia", "Zig", "Nim",
    "OCaml", "Erlang", "F#", "Groovy", "Cobol", "Fortran", "Lisp", "Scheme",
    "Solidity", "Vyper", "Racket", "Common Lisp", "Ada", "Pascal",
    "Delphi", "Objective-C", "ABAP", "SAS", "MATLAB",
    # Frameworks & web
    "Django", "Flask", "FastAPI", "React", "Vue", "Vue.js", "Angular",
    "Svelte", "Solid.js", "Qwik", "Remix", "Astro", "Next.js", "Nuxt",
    "Node.js", "Deno", "Bun", "Express", "Express.js", "Spring",
    "Spring Boot", "Laravel", "Rails", "Ruby on Rails", "ASP.NET",
    "Tailwind", "Tailwind CSS", "Bootstrap", "jQuery", "D3.js",
    "Three.js", "Chart.js", "Sass", "Less", "Stylus",
    "Jinja", "Jinja2", "Handlebars", "Mustache", "Pug",
    "Tkinter", "PyQt", "wxPython", "Electron", "Tauri",
    "WPF", "WinForms", "UWP", "MAUI", "Flutter", "React Native",
    # Infrastructure & DevOps
    "Docker", "Kubernetes", "K8s", "Terraform", "Ansible", "Puppet",
    "Chef", "SaltStack", "Jenkins", "GitHub Actions", "CircleCI",
    "Travis CI", "GitLab CI", "TeamCity", "Bamboo",
    "Prometheus", "Grafana", "Datadog", "New Relic", "Sentry",
    "OpenTelemetry", "Jaeger", "Zipkin", "Loki", "Tempo",
    "Nginx", "Apache", "Caddy", "HAProxy", "Traefik", "Envoy",
    "Istio", "Linkerd", "Consul", "Vault", "Nomad",
    "Kong", "Ambassador", "Cloudflare Workers",
    # Databases
    "SQLite", "PostgreSQL", "Postgres", "MySQL", "MariaDB", "MongoDB",
    "Redis", "Elasticsearch", "Cassandra", "DynamoDB", "CouchDB",
    "Firebase", "Supabase", "PlanetScale", "Neon", "ClickHouse",
    "DuckDB", "InfluxDB", "TimescaleDB", "CockroachDB", "Memcached",
    "BigQuery", "Snowflake", "Redshift", "Spanner", "Cosmos DB",
    "Neo4j", "ArangoDB", "OrientDB", "Dgraph", "Fauna",
    "SQL Server", "Oracle", "DB2", "SQLite",
    # ML / AI
    "TensorFlow", "PyTorch", "JAX", "scikit-learn", "Pandas", "NumPy",
    "SciPy", "Matplotlib", "Seaborn", "Plotly", "Bokeh",
    "Keras", "MXNet", "PaddlePaddle", "CNTK", "Theano", "Caffe",
    "OpenCV", "NLTK", "spaCy", "Stanford NLP", "Hugging Face",
    "Transformers", "Diffusers", "LangChain", "LlamaIndex", "Haystack",
    "Rasa", "Ollama", "vLLM", "CTransformers", "llama.cpp",
    "OpenAI", "Anthropic", "Claude", "GPT", "Gemini", "Mistral",
    "Llama", "Stable Diffusion", "Midjourney", "DALL-E",
    # Companies
    "Microsoft", "Google", "Apple", "Amazon", "Meta", "Netflix",
    "Tesla", "SpaceX", "Oracle", "IBM", "Intel", "AMD", "NVIDIA",
    "Samsung", "Sony", "Adobe", "Salesforce", "SAP", "VMware",
    "Red Hat", "Canonical", "HashiCorp", "Datadog", "Snowflake",
    "Palantir", "CrowdStrike", "Cloudflare", "GitLab", "GitHub",
    "Atlassian", "Slack", "Shopify", "Uber", "Airbnb", "Twitter/X",
    "LinkedIn", "Pinterest", "Spotify", "Stripe", "Square",
    "Twilio", "SendGrid", "MongoDB", "Elastic", "Confluent",
    "Databricks", "Hugging Face", "Anthropic", "OpenAI",
    "YOLO", "MediaPipe", "Tesseract", "EasyOCR",
    "XGBoost", "LightGBM", "CatBoost", "Random Forest",
    # Cloud providers & services
    "AWS", "GCP", "Azure", "Cloudflare", "Vercel", "Netlify",
    "Heroku", "DigitalOcean", "Linode", "Fly.io", "Railway",
    "Render", "Lambda", "Lambda Functions", "S3", "EC2",
    "CloudFront", "Route 53", "API Gateway", "CloudRun",
    "Firebase", "App Engine", "Cloud Functions",
    # Tools
    "Git", "GitHub", "GitLab", "Bitbucket", "Gitea", "Codeberg",
    "Mercurial", "SVN", "CVS",
    "VS Code", "Vim", "Neovim", "Emacs", "IntelliJ", "PyCharm",
    "WebStorm", "GoLand", "CLion", "Rider", "RubyMine",
    "Xcode", "Android Studio", "Sublime Text", "Brackets",
    "Webpack", "Vite", "Parcel", "Rollup", "esbuild", "Turbopack",
    "Babel", "SWC", "tsc", "PostCSS",
    "Jest", "Mocha", "Chai", "Sinon", "Cypress", "Playwright",
    "pytest", "Vitest", "unittest", "nose",
    "Selenium", "Cucumber", "Robot Framework", "Locust",
    "FFmpeg", "ImageMagick", "Blender", "Unity", "Unreal",
    "Make", "CMake", "Bazel", "Ninja", "Gradle", "Maven",
    "Ant", "SBT", "Leiningen",
    "Yarn", "npm", "pnpm", "Bun", "Cargo", "pip", "conda",
    "Poetry", "Pipenv", "PDM", "Rye", "uv",
    "Homebrew", "Chocolatey", "Scoop", "apt", "yum", "pacman",
    "jq", "yq", "curl", "wget", "ripgrep", "fd", "fzf",
    "bat", "exa", "lsd", "zoxide", "tmux", "screen",
    # Protocols & specs
    "GraphQL", "REST", "gRPC", "WebSocket", "MQTT", "HTTP",
    "HTTPS", "OAuth", "OAuth2", "JWT", "OpenAPI", "Swagger",
    "Protobuf", "Avro", "Parquet", "Arrow",
    "POSIX", "Systemd", "D-Bus", "Flatpak", "Snap", "AppImage",
    # Operating systems
    "Linux", "Windows", "macOS", "Android", "iOS", "FreeBSD",
    "OpenBSD", "NetBSD", "Alpine", "Ubuntu", "Debian",
    "Fedora", "RHEL", "CentOS", "Arch", "Manjaro", "NixOS",
    # Hardware / embedded
    "Raspberry Pi", "Arduino", "ESP32", "ESP8266",
    "CUDA", "ROCm", "OpenCL", "Vulkan", "DirectX",
    "OpenGL", "Metal", "WebGPU",
}

_KNOWN_FIRST_NAMES: set[str] = {
    "James", "Mary", "John", "Patricia", "Robert", "Jennifer",
    "Michael", "Linda", "David", "Barbara", "William", "Elizabeth",
    "Richard", "Susan", "Joseph", "Jessica", "Thomas", "Sarah",
    "Christopher", "Karen", "Charles", "Lisa", "Daniel", "Nancy",
    "Matthew", "Betty", "Anthony", "Margaret", "Mark", "Sandra",
    "Donald", "Ashley", "Steven", "Kimberly", "Paul", "Emily",
    "Andrew", "Donna", "Joshua", "Michelle", "Kenneth", "Carol",
    "George", "Amanda", "Edward", "Melissa", "Brian", "Deborah",
    "Ronald", "Stephanie", "Kevin", "Rebecca", "Jason", "Sharon",
    "Jeffrey", "Laura", "Ryan", "Cynthia", "Jacob", "Kathleen",
    "Gary", "Amy", "Nicholas", "Angela", "Eric", "Helen",
    "Jonathan", "Anna", "Stephen", "Brenda", "Larry", "Pamela",
    "Justin", "Samantha", "Scott", "Katherine", "Brandon", "Christine",
    "Benjamin", "Debra", "Samuel", "Rachel", "Raymond", "Carolyn",
    "Gregory", "Janet", "Frank", "Catherine", "Alexander", "Maria",
    "Patrick", "Heather", "Jack", "Diane", "Dennis", "Julie",
    "Jerry", "Joyce", "Tyler", "Evelyn", "Aaron", "Joan",
    "Jose", "Victoria", "Nathan", "Kelly", "Henry", "Lauren",
    "Douglas", "Christina", "Peter", "Judith", "Adam", "Megan",
    "Zachary", "Andrea", "Walter", "Cheryl", "Kyle", "Jacqueline",
    "Carl", "Martha", "Jeremy", "Madison", "Harold", "Teresa",
    "Keith", "Ann", "Roger", "Sara", "Gerald", "Janice",
    "Ethan", "Terry", "Lori", "Christian", "Max", "Ella", "Theo",
    "Liam", "Noah", "Oliver", "Elijah", "Lucas", "Mason", "Logan",
    "Aiden", "Carter", "Jameson", "Asher", "Silas", "Ezra", "Owen",
    "Luca", "Amelia", "Olivia", "Charlotte", "Sophia", "Isabella",
    "Mia", "Evelyn", "Harper", "Luna", "Chloe", "Penelope", "Layla",
    "Riley", "Zoey", "Nora", "Lily", "Aria", "Aurora", "Stella",
    "Mila", "Hannah", "Avery", "Levi", "Gabriel", "Isaac",
    "Muhammad", "Julian", "Mateo", "Sebastian", "Adrian",
    "Kayden", "Blake", "Bentley", "Axel", "Dominic", "Jaxon",
    "Greyson", "Holden", "Jasper", "Kai", "Kairo", "Legend",
    "Messiah", "Micah", "Oakley", "Prince", "Rowan", "Ryker",
    "Sawyer", "Tucker", "Wesley", "Zane", "Athena", "Autumn",
    "Brooklyn", "Elena", "Ellie", "Everly", "Genesis", "Hazel",
    "Iris", "Ivy", "Julia", "Kehlani", "Kennedy", "Kiara",
    "Kinsley", "Leah", "Maya", "Naomi", "Nevaeh", "Quinn",
    "Ruby", "Sage", "Savannah", "Scarlett", "Sienna", "Skylar",
    "Valentina", "Violet", "Willow", "Zara",
}

_KNOWN_SURNAMES: set[str] = {
    "Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia",
    "Miller", "Davis", "Rodriguez", "Martinez", "Hernandez", "Lopez",
    "Gonzalez", "Wilson", "Anderson", "Thomas", "Taylor", "Moore",
    "Jackson", "Martin", "Lee", "White", "Harris", "Thompson",
    "Robinson", "Clark", "Lewis", "Walker", "Hall", "Allen",
    "Young", "King", "Wright", "Hill", "Scott", "Green",
    "Adams", "Baker", "Nelson", "Carter", "Mitchell", "Roberts",
    "Turner", "Phillips", "Campbell", "Parker", "Evans", "Edwards",
    "Collins", "Stewart", "Morris", "Rogers", "Reed", "Cook",
    "Morgan", "Bell", "Murphy", "Bailey", "Rivera", "Cooper",
    "Richardson", "Cox", "Howard", "Ward", "Torres", "Peterson",
    "Gray", "Ramirez", "James", "Watson", "Brooks", "Kelly",
    "Sanders", "Price", "Bennett", "Wood", "Barnes", "Ross",
    "Henderson", "Coleman", "Jenkins", "Perry", "Powell", "Long",
    "Patterson", "Hughes", "Flores", "Washington", "Butler", "Simmons",
    "Foster", "Gonzales", "Bryant", "Alexander", "Russell", "Griffin",
    "Diaz", "Hayes", "Myers", "Ford", "Hamilton", "Graham",
    "Wallace", "Woods", "Cole", "West", "Jordan", "Owens",
    "Reynolds", "Fisher", "Ellis", "Harrison", "Gibson", "Mcdonald",
    "Cruz", "Marshall", "Mendoza", "Medina", "Fowler", "Grant",
    "Nguyen", "Tran", "Le", "Pham", "Vu", "Hoang", "Bui",
    "Vo", "Dang", "Do", "Huynh", "Truong", "Mai", "Chau",
    "Thi", "Ly", "Kim", "Phan", "Lam", "Duong", "Dinh",
    "Chen", "Wang", "Li", "Zhang", "Liu", "Yang", "Huang",
    "Wu", "Zhou", "Xu", "Sun", "Ma", "Zhu", "Hu", "Guo",
    "He", "Lin", "Luo", "Gao", "Cao", "Tang", "Han",
    "Patel", "Shah", "Kumar", "Singh", "Sharma", "Verma",
    "Gupta", "Reddy", "Rao", "Nair", "Menon", "Iyer",
    "Desai", "Joshi", "Das", "Sen", "Bose", "Ghosh",
    "Johansson", "Andersson", "Nilsson", "Karlsson",
    "Muller", "Schmidt", "Schneider", "Fischer", "Weber",
    "Wagner", "Becker", "Hoffmann", "Schäfer", "Koch",
    "Rossi", "Russo", "Ferrari", "Esposito", "Bianchi",
    "Romano", "Gallo", "Costa", "Fontana", "Conti",
    "MacDonald", "MacKenzie", "Campbell", "Stewart", "Murray",
    "Armstrong", "Crawford", "Douglas", "Ferguson", "Gibson",
    "O'Brien", "O'Sullivan", "O'Connor", "O'Neill", "Ryan",
    "Walsh", "Byrne", "Lynch", "Doyle", "McCarthy",
    "Khalil", "Haddad", "Said", "Abboud", "Nassar",
    "Yamamoto", "Sato", "Tanaka", "Watanabe", "Nakamura",
}

_KNOWN_LOCATIONS: set[str] = {
    # World cities
    "New York", "London", "Tokyo", "Paris", "Berlin", "Moscow",
    "Beijing", "Shanghai", "Dubai", "Singapore", "Sydney", "Toronto",
    "San Francisco", "Los Angeles", "Chicago", "Seattle", "Boston",
    "Amsterdam", "Barcelona", "Rome", "Madrid", "Mumbai", "Delhi",
    "Seoul", "Bangkok", "Istanbul", "Cairo", "Lagos", "Nairobi",
    "São Paulo", "Rio de Janeiro", "Buenos Aires", "Mexico City",
    "Miami", "Dallas", "Austin", "Portland", "Denver", "Atlanta",
    "Phoenix", "Houston", "Philadelphia", "San Diego", "Minneapolis",
    "Detroit", "Memphis", "Baltimore", "Cleveland", "Pittsburgh",
    "Cincinnati", "Nashville", "Kansas City", "Milwaukee",
    "New Orleans", "Salt Lake City", "Las Vegas", "Orlando",
    "Charlotte", "Raleigh", "Indianapolis", "Columbus", "Sacramento",
    "San Antonio", "San Jose", "Oakland", "St. Louis", "Tampa",
    "Vancouver", "Montreal", "Calgary", "Ottawa", "Edmonton",
    "Hong Kong", "Kuala Lumpur", "Jakarta", "Manila",
    "Ho Chi Minh City", "Hanoi", "Taipei", "Osaka", "Kyoto",
    "Wellington", "Auckland", "Christchurch",
    "Copenhagen", "Stockholm", "Oslo", "Helsinki",
    "Zurich", "Geneva", "Munich", "Hamburg", "Frankfurt",
    "Brussels", "Antwerp", "Vienna", "Prague", "Budapest",
    "Warsaw", "Krakow", "Dublin", "Cork", "Lisbon", "Porto",
    "Athens", "Thessaloniki", "Tel Aviv", "Jerusalem", "Haifa",
    "Riyadh", "Jeddah", "Doha", "Abu Dhabi", "Kuwait City",
    "Kolkata", "Bangalore", "Hyderabad", "Chennai", "Pune",
    "Ahmedabad", "Jaipur", "Lucknow", "Surat",
    "Lahore", "Karachi", "Islamabad", "Faisalabad",
    "Dhaka", "Chittagong", "Casablanca", "Marrakesh", "Rabat",
    "Cape Town", "Johannesburg", "Durban", "Pretoria",
    "Addis Ababa", "Nairobi", "Accra", "Kumasi",
    # Countries
    "United States", "USA", "United States of America",
    "Canada", "Mexico", "Brazil", "Argentina", "Chile", "Colombia",
    "Peru", "Venezuela", "Uruguay", "Paraguay", "Bolivia", "Ecuador",
    "United Kingdom", "UK", "Great Britain", "England", "Scotland",
    "Wales", "Northern Ireland",
    "Germany", "France", "Italy", "Spain", "Portugal",
    "Netherlands", "Belgium", "Switzerland", "Austria",
    "Sweden", "Norway", "Denmark", "Finland", "Iceland",
    "Poland", "Czech Republic", "Slovakia", "Hungary",
    "Romania", "Bulgaria", "Serbia", "Croatia", "Slovenia",
    "Greece", "Turkey", "Cyprus",
    "Russia", "Ukraine", "Belarus", "Lithuania", "Latvia",
    "Estonia", "Georgia", "Armenia", "Azerbaijan",
    "China", "Japan", "South Korea", "North Korea", "Mongolia",
    "India", "Pakistan", "Bangladesh", "Sri Lanka", "Nepal",
    "Bhutan", "Myanmar", "Laos", "Cambodia", "Vietnam",
    "Thailand", "Malaysia", "Indonesia", "Philippines",
    "Singapore", "Brunei", "East Timor",
    "Australia", "New Zealand", "Papua New Guinea", "Fiji",
    "Saudi Arabia", "UAE", "United Arab Emirates", "Qatar",
    "Kuwait", "Oman", "Bahrain", "Jordan", "Lebanon",
    "Israel", "Palestine", "Syria", "Iraq", "Iran", "Yemen",
    "Egypt", "Libya", "Tunisia", "Algeria", "Morocco", "Sudan",
    "South Africa", "Nigeria", "Ghana", "Kenya", "Ethiopia",
    "Tanzania", "Uganda", "Rwanda", "DR Congo", "Angola",
    "Mozambique", "Zambia", "Zimbabwe", "Botswana", "Namibia",
    "Senegal", "Mali", "Burkina Faso", "Cameroon", "Ivory Coast",
    # Regions & states
    "Silicon Valley", "Bay Area", "Wall Street", "Midwest",
    "West Coast", "East Coast", "Gulf Coast",
    "Southeast Asia", "Western Europe", "Eastern Europe",
    "Middle East", "Central America", "South America",
    "North America", "Latin America", "Scandinavia",
    "Nordic", "Balkans", "Baltics", "Benelux",
    "California", "Texas", "Florida", "New York State",
    "Nevada", "Oregon", "Washington State", "Colorado",
    "Arizona", "Illinois", "Massachusetts", "Pennsylvania",
    "Georgia", "North Carolina", "Michigan", "Ohio",
    "Virginia", "Washington DC", "Hawaii", "Alaska",
    "Bavaria", "Catalonia", "Quebec", "Ontario",
    "British Columbia", "Tuscany", "Provence", "Burgundy",
    "Great Lakes", "Amazon Rainforest", "Sahara Desert",
}

_KNOWN_NAMES: set[str] = {
    "Alice", "Bob", "Charlie", "David", "Eve", "Frank", "Grace",
    "Hank", "Ivy", "Jack", "Kate", "Leo", "Mallory", "Nina",
    "Oscar", "Peggy", "Quinn", "Ruth", "Sam", "Trent", "Uma",
    "Victor", "Wendy", "Xavier", "Yvonne", "Zack",
}

_ALL_KNOWN_NAMES: set[str] = _KNOWN_NAMES | _KNOWN_FIRST_NAMES | _KNOWN_SURNAMES
_ALL_KNOWN: set[str] = _ALL_KNOWN_NAMES | _KNOWN_TECHNOLOGIES | _KNOWN_LOCATIONS

_TITLES: list[str] = [
    "Dr", "Prof", "Mr", "Ms", "Mrs", "Sir", "Lady", "Lord",
    "Capt", "Sgt", "Rep", "Sen", "Gov", "Pres", "VP",
]

_TITLE_SET: set[str] = set(_TITLES)

# Words that break capitalized word sequences (prevent grouping)
_SEQUENCE_BREAKERS: set[str] = {
    "The", "This", "That", "These", "Those",
    "Using", "Building", "Creating", "Working", "Running", "Testing",
    "Developing", "Deploying", "Installing", "Configuring", "Setting",
    "Getting", "Putting", "Making", "Doing", "Going", "Taking",
    "Adding", "Removing", "Updating", "Fixing", "Starting", "Stopping",
    "However", "Therefore", "Furthermore", "Moreover", "Additionally",
    "Also", "Even", "Just", "Only", "Really", "Actually",
}

_ENTITY_STOPLIST: set[str] = {
    "The", "This", "That", "These", "Those",
    "I", "You", "He", "She", "It", "We", "They",
    "My", "Your", "His", "Her", "Its", "Our", "Their",
    "Mine", "Yours", "Hers", "Ours", "Theirs",
    "A", "An", "And", "But", "Or", "For", "Nor", "Yet", "So",
    "If", "Then", "Else", "When", "Where", "Why", "How",
    "Which", "What", "Who", "Whom", "Whose",
    "All", "Each", "Every", "Both", "Few", "Many", "Much",
    "Some", "Any", "No", "Not", "None", "Nothing",
    "Only", "Just", "Also", "Very", "Too", "Here", "There",
    "Please", "Hello", "Hi", "Hey", "Goodbye", "Bye",
    "Thanks", "Thank", "Please",
    # Months
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
    # Days
    "Monday", "Tuesday", "Wednesday", "Thursday", "Friday",
    "Saturday", "Sunday",
    # Generic
    "Yesterday", "Today", "Tomorrow",
    "User", "Assistant", "System", "Human", "Bot",
    "Note", "Warning", "Error", "Info", "Debug",
    "Todo", "FIXME", "HACK", "XXX",
    "Yes", "No", "True", "False", "None", "Null", "NaN",
    "Type", "Value", "Key", "Name", "File", "Line",
    "Up", "Down", "Left", "Right", "On", "Off",
    "First", "Second", "Third", "Last", "Next", "Previous",
    "Top", "Bottom", "Middle", "Center",
    "New", "Old", "Good", "Bad", "Great", "Little",
    "Big", "Large", "Small", "High", "Low", "Long", "Short",
    "Once", "Twice", "Often", "Always", "Never", "Sometimes",
    "Please", "Sorry", "Hello", "Hi",
    # Title words (caught by specific patterns instead)
    "Dr", "Prof", "Mr", "Ms", "Mrs", "Sir", "Lady", "Lord",
    "Capt", "Sgt", "Rep", "Sen", "Gov", "Pres", "VP",
    # Gerunds & verbs commonly at sentence start
    "Using", "Building", "Creating", "Working", "Running", "Testing",
    "Developing", "Deploying", "Installing", "Configuring", "Setting",
    "Getting", "Putting", "Making", "Doing", "Going", "Taking",
    "Adding", "Removing", "Updating", "Fixing", "Starting", "Stopping",
}


class EntityDetector:
    """Detects named entities in text using regex patterns and heuristics."""

    def __init__(self) -> None:
        self._load_patterns()

    def _load_patterns(self) -> None:
        title_alt = "|".join(_TITLES)
        self._title_person_re = re.compile(
            rf"(?i:(?:{title_alt}))\.?\s+"
            rf"(?-i:([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?))",
        )

        self._project_re = re.compile(
            r"""(?i:(?:project|initiative|repo:?|codebase|package|library|tool|"""
            r"""utility|framework|platform|engine|module))\s+"""
            r"""(?-i:["']?([A-Z][A-Za-z0-9_-]+(?:\s+[A-Z][A-Za-z0-9_-]+)?))""",
        )

        self._tech_trigger_re = re.compile(
            r"""(?i:(?:using|built with|written in|powered by|runs on|"""
            r"""implemented in|built on top of|based on|"""
            r"""developed with|created with|made with))\s+"""
            r"""(?-i:["']?([A-Z][A-Za-z0-9+#.]*(?:\s+[A-Z][A-Za-z0-9+#.]*){0,2}))""",
        )

        self._org_prep_re = re.compile(
            r"(?i:(?:at|by|for|with|from))\s+"
            r"(?-i:([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,}))",
        )

        self._loc_prep_re = re.compile(
            r"(?i:(?:in|to))\s+"
            r"(?-i:([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?))",
        )

        self._iso_date_re = re.compile(
            r"\b(\d{4}-\d{2}-\d{2}(?:T\d{2}:\d{2}:\d{2}(?:\.\d+)?Z?)?)\b",
        )

        self._relative_date_re = re.compile(
            r"(?i)\b(yesterday|today|tomorrow|"
            r"last\s+(week|month|year|monday|tuesday|wednesday|"
            r"thursday|friday|saturday|sunday)|"
            r"next\s+(week|month|year|monday|tuesday|wednesday|"
            r"thursday|friday|saturday|sunday)|"
            r"this\s+(week|month|year))\b",
        )

        self._event_marker_re = re.compile(
            r"""(?i:(?:event|meeting|conference|talk|workshop|seminar|"""
            r"""webinar|hackathon|summit|symposium|keynote|"""
            r"""lecture|panel|forum|retreat)):\s*"""
            r"""(?-i:["']?([A-Z][A-Za-z0-9]*(?:\s+[A-Z][A-Za-z0-9]*)*))""",
        )

        self._cap_word_re = re.compile(r"\b([A-Z][a-z]+)\b")

        self._capital_pascal_re = re.compile(r"\b([A-Z][a-z0-9]+(?:[A-Z][a-z0-9]+)+)\b")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect(self, text: str) -> list[dict[str, Any]]:
        """Detect all entities in text.

        Returns list of dicts:
          {name: str, type: str, confidence: float, position: (start, end)}
          Types: person, project, location, organization, technology, concept, event, date
        """
        if not text or not text.strip():
            return []

        candidates: list[dict[str, Any]] = []

        # --- Pattern-based detection (highest confidence) ---

        self._detect_title_names(text, candidates)
        self._detect_project_mentions(text, candidates)
        self._detect_tech_triggers(text, candidates)
        self._detect_org_prepositions(text, candidates)
        self._detect_loc_prepositions(text, candidates)
        self._detect_dates(text, candidates)
        self._detect_event_markers(text, candidates)
        self._detect_pascal_case(text, candidates)

        # --- General capitalized sequence scanning ---

        self._scan_capitalized_sequences(text, candidates)

        return self._deduplicate(candidates)

    # ------------------------------------------------------------------
    # Individual detectors
    # ------------------------------------------------------------------

    def _detect_title_names(self, text: str, candidates: list[dict]) -> None:
        for m in self._title_person_re.finditer(text):
            name = m.group(1).strip()
            start, end = m.start(1), m.end(1)
            if name in _ENTITY_STOPLIST:
                continue
            candidates.append(self._make(name, "person", 0.92, start, end))

    def _detect_project_mentions(self, text: str, candidates: list[dict]) -> None:
        for m in self._project_re.finditer(text):
            name = m.group(1).strip()
            start, end = m.start(1), m.end(1)
            if name in _ENTITY_STOPLIST:
                continue
            if name in _KNOWN_TECHNOLOGIES:
                candidates.append(self._make(name, "technology", 0.9, start, end))
            elif name in _ALL_KNOWN_NAMES:
                candidates.append(self._make(name, "person", 0.8, start, end))
            elif name in _KNOWN_LOCATIONS:
                candidates.append(self._make(name, "location", 0.8, start, end))
            else:
                candidates.append(self._make(name, "project", 0.85, start, end))

    def _detect_tech_triggers(self, text: str, candidates: list[dict]) -> None:
        for m in self._tech_trigger_re.finditer(text):
            name = m.group(1).strip()
            start, end = m.start(1), m.end(1)
            if name in _ENTITY_STOPLIST:
                continue
            if name in _KNOWN_TECHNOLOGIES:
                candidates.append(self._make(name, "technology", 0.93, start, end))
            elif name in _ALL_KNOWN_NAMES:
                candidates.append(self._make(name, "person", 0.7, start, end))
            elif name in _KNOWN_LOCATIONS:
                candidates.append(self._make(name, "location", 0.7, start, end))
            else:
                candidates.append(self._make(name, "technology", 0.72, start, end))

    def _detect_org_prepositions(self, text: str, candidates: list[dict]) -> None:
        for m in self._org_prep_re.finditer(text):
            name = m.group(1).strip()
            start, end = m.start(1), m.end(1)
            if name in _ENTITY_STOPLIST:
                continue
            if name in _KNOWN_TECHNOLOGIES:
                candidates.append(self._make(name, "technology", 0.8, start, end))
            elif name in _KNOWN_LOCATIONS:
                candidates.append(self._make(name, "location", 0.8, start, end))
            else:
                candidates.append(self._make(name, "organization", 0.75, start, end))

    def _detect_loc_prepositions(self, text: str, candidates: list[dict]) -> None:
        for m in self._loc_prep_re.finditer(text):
            name = m.group(1).strip()
            start, end = m.start(1), m.end(1)
            if name in _ENTITY_STOPLIST:
                continue
            if name in _KNOWN_LOCATIONS:
                candidates.append(self._make(name, "location", 0.9, start, end))
            elif name in _KNOWN_TECHNOLOGIES:
                candidates.append(self._make(name, "technology", 0.7, start, end))
            elif name in _ALL_KNOWN_NAMES:
                candidates.append(self._make(name, "person", 0.7, start, end))
            else:
                candidates.append(self._make(name, "location", 0.7, start, end))

    def _detect_dates(self, text: str, candidates: list[dict]) -> None:
        for m in self._iso_date_re.finditer(text):
            candidates.append(
                self._make(m.group(1), "date", 0.95, m.start(1), m.end(1)),
            )
        for m in self._relative_date_re.finditer(text):
            candidates.append(
                self._make(m.group(1).lower(), "date", 0.85, m.start(1), m.end(1)),
            )

    def _detect_event_markers(self, text: str, candidates: list[dict]) -> None:
        for m in self._event_marker_re.finditer(text):
            name = m.group(1).strip()
            start, end = m.start(1), m.end(1)
            if not name or name in _ENTITY_STOPLIST:
                continue
            candidates.append(self._make(name, "event", 0.9, start, end))

    def _detect_pascal_case(self, text: str, candidates: list[dict]) -> None:
        for m in self._capital_pascal_re.finditer(text):
            name = m.group(1).strip()
            start, end = m.start(1), m.end(1)
            if name in _ENTITY_STOPLIST or name in _ALL_KNOWN:
                continue
            if len(name) >= 4:
                candidates.append(self._make(name, "project", 0.7, start, end))

    # ------------------------------------------------------------------
    # Capitalized-sequence scanning
    # ------------------------------------------------------------------

    def _scan_capitalized_sequences(self, text: str, candidates: list[dict]) -> None:
        """Find capitalized word runs and classify them."""
        cap_matches = list(self._cap_word_re.finditer(text))
        if not cap_matches:
            return

        sequences = self._build_capitalized_sequences(cap_matches)

        for seq in sequences:
            words = seq["words"]
            seq_start = seq["start"]
            seq_end = seq["end"]
            name = " ".join(words)

            if name in _ENTITY_STOPLIST:
                continue

            length = len(words)
            at_sentence_start = self._is_sentence_start(seq_start, text)

            # Check known-entity lookups first (these override stoplist membership)
            if name in _KNOWN_TECHNOLOGIES:
                candidates.append(
                    self._make(name, "technology", 0.95, seq_start, seq_end),
                )
                continue
            if name in _KNOWN_LOCATIONS:
                candidates.append(
                    self._make(name, "location", 0.95, seq_start, seq_end),
                )
                continue

            at_sentence_start = self._is_sentence_start(seq_start, text)

            if length >= 3:
                candidates.append(
                    self._make(name, "organization", 0.7, seq_start, seq_end),
                )

            elif length == 2:
                w1, w2 = words
                if (w1 in _KNOWN_FIRST_NAMES and w2 in _KNOWN_SURNAMES) or (w1 in _ALL_KNOWN_NAMES and w2 in _ALL_KNOWN_NAMES):
                    candidates.append(
                        self._make(name, "person", 0.88, seq_start, seq_end),
                    )
                elif at_sentence_start:
                    candidates.append(
                        self._make(name, "concept", 0.55, seq_start, seq_end),
                    )
                else:
                    candidates.append(
                        self._make(name, "person", 0.65, seq_start, seq_end),
                    )

            else:  # single word
                word = words[0]
                if word in _ALL_KNOWN_NAMES:
                    if at_sentence_start:
                        candidates.append(
                            self._make(word, "person", 0.6, seq_start, seq_end),
                        )
                    else:
                        candidates.append(
                            self._make(word, "person", 0.65, seq_start, seq_end),
                        )
                elif word in _KNOWN_TECHNOLOGIES:
                    candidates.append(
                        self._make(word, "technology", 0.9, seq_start, seq_end),
                    )
                elif word in _KNOWN_LOCATIONS:
                    candidates.append(
                        self._make(word, "location", 0.9, seq_start, seq_end),
                    )
                elif at_sentence_start:
                    continue
                else:
                    candidates.append(
                        self._make(word, "person", 0.5, seq_start, seq_end),
                    )

    def _build_capitalized_sequences(
        self, matches: list[re.Match],
    ) -> list[dict[str, Any]]:
        """Group consecutive capitalized words into multi-word sequences."""
        if not matches:
            return []

        sorted_matches = sorted(matches, key=lambda m: m.start())

        sequences: list[dict[str, Any]] = []
        current_run: list[re.Match] = [sorted_matches[0]]

        for i in range(1, len(sorted_matches)):
            prev_end = sorted_matches[i - 1].end()
            curr_start = sorted_matches[i].start()
            gap = curr_start - prev_end
            word = sorted_matches[i].group(1)
            if gap <= 1 and word not in _SEQUENCE_BREAKERS:
                current_run.append(sorted_matches[i])
            else:
                sequences.append(current_run)
                current_run = [sorted_matches[i]]

        if current_run:
            sequences.append(current_run)

        result: list[dict[str, Any]] = []
        for run in sequences:
            run_words = [m.group(1) for m in run]
            run_start = run[0].start()
            run_end = run[-1].end()
            n = len(run_words)

            result.append({
                "words": run_words,
                "start": run_start,
                "end": run_end,
            })

            if n >= 2:
                for i, m in enumerate(run):
                    result.append({
                        "words": [run_words[i]],
                        "start": m.start(),
                        "end": m.end(),
                    })

        return result

    # ------------------------------------------------------------------
    # Deduplication
    # ------------------------------------------------------------------

    def _deduplicate(self, candidates: list[dict]) -> list[dict[str, Any]]:
        """Remove overlapping and duplicate entities.

        1. Sort by span length descending (prefer longer spans).
        2. Remove overlapping position entries, keeping longest span.
        3. Group by normalized name, keep highest confidence.
        """
        if not candidates:
            return []

        candidates.sort(
            key=lambda e: (-e["confidence"], -(e["position"][1] - e["position"][0])),
        )

        non_overlapping: list[dict] = []
        for c in candidates:
            overlaps = False
            for kept in non_overlapping:
                if self._positions_overlap(c["position"], kept["position"]):
                    overlaps = True
                    break
            if not overlaps:
                non_overlapping.append(c)

        best: dict[str, dict] = {}
        for c in non_overlapping:
            key = c["name"].lower().strip()
            if key not in best or c["confidence"] > best[key]["confidence"]:
                best[key] = c

        result = sorted(best.values(), key=lambda e: e["position"][0])
        return result

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _make(
        name: str, type_: str, confidence: float, start: int, end: int,
    ) -> dict[str, Any]:
        return {
            "name": name,
            "type": type_,
            "confidence": round(confidence, 3),
            "position": (start, end),
        }

    @staticmethod
    def _is_sentence_start(pos: int, text: str) -> bool:
        """Check if position is at the start of a sentence."""
        if pos <= 1:
            return True
        before = text[max(0, pos - 3):pos]
        stripped = before.rstrip()
        if not stripped:
            return True
        return stripped[-1] in ".!?\n"

    @staticmethod
    def _positions_overlap(a: tuple[int, int], b: tuple[int, int]) -> bool:
        return a[0] < b[1] and b[0] < a[1]

    def confirm_entities(
        self, text: str, registry: Optional["EntityRegistry"] = None,
    ) -> tuple[list[dict], list[dict]]:
        """Interactively confirm detected entities.

        Presents each detected entity to the user, asking for confirmation,
        rejection, or renaming. Returns (confirmed, rejected).

        When *registry* is provided, confirmed entities are registered
        immediately via ``registry.register()``.
        """
        detected = self.detect(text)
        if not detected:
            print("  (no entities detected)")
            return [], []

        confirmed: list[dict] = []
        rejected: list[dict] = []

        print(f"\n  Detected {len(detected)} candidate entities:\n")
        for i, entity in enumerate(detected, 1):
            name = entity["name"]
            etype = entity["type"]
            conf = entity["confidence"]
            print(
                f"  [{i}] {name:<30s} type={etype:<12s} "
                f"conf={conf:.2f} "
                f'pos={entity["position"]}'
            )

        print()
        while True:
            try:
                raw = input("  Confirm indices (e.g. 1,3,5), range (1-3), "
                            "reject (r1,r3), rename (rn2=NewName), or blank to finish: ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break

            if not raw:
                break

            for token in raw.split(","):
                token = token.strip()
                if not token:
                    continue

                # rename: rn2=NewName
                if token.startswith("rn") and "=" in token:
                    try:
                        idx_str, new_name = token[2:].split("=", 1)
                        idx = int(idx_str.strip()) - 1
                        new_name = new_name.strip()
                        if 0 <= idx < len(detected) and new_name:
                            detected[idx]["name"] = new_name
                            confirmed.append(detected[idx])
                            print(f"    → confirmed as '{new_name}'")
                    except (ValueError, IndexError):
                        print(f"    ? invalid rename token: {token}")
                    continue

                # reject: r3
                if token.startswith("r"):
                    try:
                        idx = int(token[1:]) - 1
                        if 0 <= idx < len(detected):
                            rejected.append(detected[idx])
                            print(f"    → rejected [{token[1:]}] {detected[idx]['name']}")
                    except (ValueError, IndexError):
                        print(f"    ? invalid reject token: {token}")
                    continue

                # range: 1-3
                if "-" in token:
                    try:
                        a, b = token.split("-", 1)
                        for idx in range(int(a) - 1, int(b)):
                            if 0 <= idx < len(detected) and detected[idx] not in confirmed:
                                confirmed.append(detected[idx])
                    except (ValueError, IndexError):
                        print(f"    ? invalid range: {token}")
                    continue

                # single index
                try:
                    idx = int(token) - 1
                    if 0 <= idx < len(detected):
                        if detected[idx] not in confirmed:
                            confirmed.append(detected[idx])
                            print(f"    → confirmed [{token}] {detected[idx]['name']}")
                except ValueError:
                    print(f"    ? invalid token: {token}")

        if registry:
            for entity in confirmed:
                registry.register(entity["name"], source="user-confirmed")

        return confirmed, rejected


# ==================== MODULE-LEVEL ENTITY DETECTION (i18n multi-language) ====================


def _normalize_langs(languages) -> tuple:
    if not languages:
        return ("en",)
    if isinstance(languages, str):
        return (languages,)
    return tuple(languages)


@functools.lru_cache(maxsize=32)
def _get_stopwords(languages: tuple) -> frozenset:
    patterns = get_entity_patterns(languages)
    return frozenset(patterns["stopwords"])


def extract_candidates(text: str, languages=("en",)) -> dict:
    langs = _normalize_langs(languages)
    patterns = get_entity_patterns(langs)
    stopwords = _get_stopwords(langs)
    coca_filter = _get_coca_filter()

    counts: defaultdict = defaultdict(int)

    working_text, compound_counts = _apply_known_systems_prepass(text)
    for compound, n in compound_counts.items():
        counts[compound] += n

    for wrapped_pat in patterns["candidate_patterns"]:
        try:
            rx = re.compile(wrapped_pat)
        except re.error:
            continue
        for word in rx.findall(working_text):
            wl = word.lower()
            if wl in stopwords:
                continue
            if wl in coca_filter:
                continue
            if len(word) < 2:
                continue
            counts[word] += 1

    for wrapped_pat in patterns["multi_word_patterns"]:
        try:
            rx = re.compile(wrapped_pat)
        except re.error:
            continue
        for phrase in rx.findall(working_text):
            if any(w.lower() in stopwords for w in phrase.split()):
                continue
            counts[phrase] += 1

    return {name: count for name, count in counts.items() if count >= 3}


def score_entity(name: str, text: str, lines: list, languages=("en",)) -> dict:
    langs = _normalize_langs(languages)
    n = re.escape(name)
    patterns = get_entity_patterns(langs)

    def _compile(raw_patterns, flags=re.IGNORECASE):
        compiled = []
        for p in raw_patterns:
            try:
                compiled.append(re.compile(p.format(name=n), flags))
            except (re.error, KeyError, IndexError):
                continue
        return compiled

    dialogue_rx = _compile(patterns["dialogue_patterns"], re.MULTILINE | re.IGNORECASE)
    person_verb_rx = _compile(patterns["person_verb_patterns"])
    project_verb_rx = _compile(patterns["project_verb_patterns"])

    direct_raw = patterns.get("direct_address_patterns") or []
    direct_rx = []
    for raw in direct_raw:
        try:
            direct_rx.append(re.compile(raw.format(name=n), re.IGNORECASE))
        except (re.error, KeyError, IndexError):
            continue

    versioned_rx = re.compile(rf"\b{n}[-_]v?\d+(?:\.\d+)*\b", re.IGNORECASE)
    code_ref_rx = re.compile(rf"\b{n}\.(py|js|ts|yaml|yml|json|sh)\b", re.IGNORECASE)

    pronouns = patterns.get("pronoun_patterns") or []
    pronoun_re = None
    if pronouns:
        try:
            pronoun_re = re.compile("|".join(pronouns), re.IGNORECASE)
        except re.error:
            pass

    person_score = 0
    project_score = 0
    person_signals = []
    project_signals = []

    for rx in dialogue_rx:
        matches = len(rx.findall(text))
        if matches == 0:
            continue
        is_bare_colon = rx.pattern.endswith(r":\s") and not rx.pattern.endswith(r"[:\s]")
        if is_bare_colon and matches < 2:
            continue
        person_score += matches * 3
        person_signals.append(f"dialogue marker ({matches}x)")

    for rx in person_verb_rx:
        matches = len(rx.findall(text))
        if matches > 0:
            person_score += matches * 2
            person_signals.append(f"'{name} ...' action ({matches}x)")

    if pronoun_re is not None:
        name_lower = name.lower()
        name_line_indices = [i for i, line in enumerate(lines) if name_lower in line.lower()]
        pronoun_hits = 0
        for idx in name_line_indices:
            window_text = " ".join(lines[max(0, idx - 2): idx + 3])
            if pronoun_re.search(window_text):
                pronoun_hits += 1
        if pronoun_hits > 0:
            person_score += pronoun_hits * 2
            person_signals.append(f"pronoun nearby ({pronoun_hits}x)")

    direct_hits = 0
    for rx in direct_rx:
        direct_hits += len(rx.findall(text))
    if direct_hits > 0:
        person_score += direct_hits * 4
        person_signals.append(f"addressed directly ({direct_hits}x)")

    for rx in project_verb_rx:
        matches = len(rx.findall(text))
        if matches > 0:
            project_score += matches * 2
            project_signals.append(f"project verb ({matches}x)")

    versioned = len(versioned_rx.findall(text))
    if versioned > 0:
        project_score += versioned * 3
        project_signals.append(f"versioned/hyphenated ({versioned}x)")

    code_ref = len(code_ref_rx.findall(text))
    if code_ref > 0:
        project_score += code_ref * 3
        project_signals.append(f"code file reference ({code_ref}x)")

    return {
        "person_score": person_score,
        "project_score": project_score,
        "person_signals": person_signals[:3],
        "project_signals": project_signals[:3],
    }


def classify_entity(name: str, frequency: int, scores: dict) -> dict:
    ps = scores["person_score"]
    prs = scores["project_score"]
    total = ps + prs

    if total == 0:
        confidence = min(0.4, frequency / 50)
        return {
            "name": name,
            "type": "uncertain",
            "confidence": round(confidence, 2),
            "frequency": frequency,
            "signals": [f"appears {frequency}x, no strong type signals"],
        }

    person_ratio = ps / total

    signal_categories = set()
    for s in scores["person_signals"]:
        if "dialogue" in s:
            signal_categories.add("dialogue")
        elif "action" in s:
            signal_categories.add("action")
        elif "pronoun" in s:
            signal_categories.add("pronoun")
        elif "addressed" in s:
            signal_categories.add("addressed")

    has_two_signal_types = len(signal_categories) >= 2
    pronoun_hits = 0
    for s in scores["person_signals"]:
        m = re.search(r"pronoun nearby \((\d+)x\)", s)
        if m:
            pronoun_hits = int(m.group(1))
            break
    strong_pronoun_signal = pronoun_hits >= 5 and frequency > 0 and pronoun_hits / frequency >= 0.2

    if person_ratio >= 0.7 and (has_two_signal_types and ps >= 5 or strong_pronoun_signal):
        entity_type = "person"
        confidence = min(0.99, 0.5 + person_ratio * 0.5)
        signals = scores["person_signals"] or [f"appears {frequency}x"]
    elif person_ratio >= 0.7:
        entity_type = "uncertain"
        confidence = 0.4
        signals = scores["person_signals"] + [f"appears {frequency}x — weak person signal"]
    elif person_ratio <= 0.3:
        entity_type = "project"
        confidence = min(0.99, 0.5 + (1 - person_ratio) * 0.5)
        signals = scores["project_signals"] or [f"appears {frequency}x"]
    else:
        entity_type = "uncertain"
        confidence = 0.5
        signals = (scores["person_signals"] + scores["project_signals"])[:3]
        signals.append("mixed signals — needs review")

    return {
        "name": name,
        "type": entity_type,
        "confidence": round(confidence, 2),
        "frequency": frequency,
        "signals": signals,
    }


def detect_entities(
    file_paths: list,
    max_files: int = 10,
    languages=("en",),
    corpus_origin: dict | None = None,
) -> dict:
    langs = _normalize_langs(languages)

    all_text = []
    all_lines = []
    files_read = 0
    MAX_BYTES_PER_FILE = 5000

    for filepath in file_paths:
        if files_read >= max_files:
            break
        try:
            with open(filepath, encoding="utf-8", errors="replace") as f:
                content = f.read(MAX_BYTES_PER_FILE)
            all_text.append(content)
            all_lines.extend(content.splitlines())
            files_read += 1
        except OSError:
            continue

    combined_text = "\n".join(all_text)
    candidates = extract_candidates(combined_text, languages=langs)

    if not candidates:
        return {"people": [], "projects": [], "topics": [], "uncertain": []}

    people = []
    projects = []
    uncertain = []

    for name, frequency in sorted(candidates.items(), key=lambda x: x[1], reverse=True):
        scores = score_entity(name, combined_text, all_lines, languages=langs)
        entity = classify_entity(name, frequency, scores)
        if entity["type"] == "person":
            people.append(entity)
        elif entity["type"] == "project":
            projects.append(entity)
        else:
            uncertain.append(entity)

    people.sort(key=lambda x: x["confidence"], reverse=True)
    projects.sort(key=lambda x: x["confidence"], reverse=True)
    uncertain.sort(key=lambda x: x["frequency"], reverse=True)

    detected = {
        "people": people[:15],
        "projects": projects[:10],
        "topics": [],
        "uncertain": uncertain[:8],
    }

    if corpus_origin is not None:
        from kai_mempalace.entity_detector import _apply_corpus_origin
        detected = _apply_corpus_origin(detected, corpus_origin)

    return detected


def scan_for_detection(project_dir: str, max_files: int = 10) -> list:
    project_path = Path(project_dir).expanduser().resolve()
    prose_files = []
    all_files = []

    for root, dirs, filenames in os.walk(project_path):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
        for filename in filenames:
            filepath = Path(root) / filename
            ext = filepath.suffix.lower()
            if ext in PROSE_EXTENSIONS:
                prose_files.append(filepath)
            elif ext in {".py", ".js", ".ts", ".json", ".yaml", ".yml", ".rst", ".toml", ".sh", ".rb", ".go", ".rs"}:
                all_files.append(filepath)

    files = prose_files if len(prose_files) >= 3 else prose_files + all_files
    return files[:max_files]
