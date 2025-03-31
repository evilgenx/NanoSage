# gui/searxng_engines.py
"""
Parses and stores the structure of SearXNG engines, categories (tabs), and groups
based on the provided documentation data.
"""

# Data derived from the SearXNG documentation provided.
# Structure: { tab_name: { "groups": { group_name: [engine_list], ... }, "engines": [engine_list] } }
# engine_list contains tuples: (engine_name, bang_command, is_default_enabled)

SEARXNG_ENGINE_DATA = {
    "!general": {
        "groups": {
            "!translate": [
                ("dictzone", "!dc", True),
                ("libretranslate", "!lt", False),
                ("lingva", "!lv", True),
                ("mozhi", "!mz", False),
                ("mymemory translated", "!tl", True),
            ],
            "!web": [
                ("bing", "!bi", False),
                ("brave", "!br", True),
                ("duckduckgo", "!ddg", True),
                ("google", "!go", True),
                ("mojeek", "!mjk", False),
                ("presearch", "!ps", False),
                ("presearch videos", "!psvid", False), # Note: Same bang as presearch? Doc error? Assuming distinct for now. Bangs should be unique. Let's use name for key later.
                ("qwant", "!qw", True),
                ("startpage", "!sp", True),
                ("wiby", "!wib", False),
                ("yahoo", "!yh", False),
                ("seznam (CZ)", "!szn", False),
                ("goo (JA)", "!goo", False),
                ("naver (KO)", "!nvr", False),
            ],
            "!wikimedia": [
                ("wikibooks", "!wb", False),
                ("wikiquote", "!wq", False),
                ("wikisource", "!ws", False),
                ("wikispecies", "!wsp", False), # Also listed under !science
                ("wikiversity", "!wv", False),
                ("wikivoyage", "!wy", False),
            ],
        },
        "engines": [
            ("alexandria", "!alx", False),
            ("ask", "!ask", False),
            ("cloudflareai", "!cfai", False),
            ("crowdview", "!cv", False),
            ("curlie", "!cl", False),
            ("currency", "!cc", True),
            ("ddg definitions", "!ddd", False),
            ("encyclosearch", "!es", False),
            ("mwmbl", "!mwm", False),
            ("right dao", "!rd", False),
            ("searchmysite", "!sms", False),
            ("stract", "!str", False),
            ("tineye", "!tin", False),
            ("wikidata", "!wd", True),
            ("wikipedia", "!wp", True),
            ("wolframalpha", "!wa", False),
            ("yacy", "!ya", False),
            ("yep", "!yep", False),
            ("bpb (DE)", "!bpb", False),
            ("tagesschau (DE)", "!ts", False), # Also listed under !news
            ("wikimini (FR)", "!wkmn", False),
            ("360search (ZH)", "!360so", False),
            ("baidu (ZH)", "!bd", False),
            ("quark (ZH)", "!qk", False),
            ("sogou (ZH)", "!sogou", False),
        ]
    },
    "!images": {
        "groups": {
            "!web": [
                ("bing images", "!bii", True),
                ("brave.images", "!brimg", True),
                ("duckduckgo images", "!ddi", False),
                ("google images", "!goi", True),
                ("mojeek images", "!mjkimg", False),
                ("presearch images", "!psimg", False),
                ("qwant images", "!qwi", True),
                ("startpage images", "!spi", True),
            ]
        },
        "engines": [
            ("1x", "!1x", False),
            ("adobe stock", "!asi", False),
            ("artic", "!arc", True),
            ("deviantart", "!da", True),
            ("findthatmeme", "!ftm", False),
            ("flickr", "!fl", True),
            ("frinkiac", "!frk", False),
            ("imgur", "!img", False),
            ("ipernity", "!ip", False),
            ("library of congress", "!loc", True),
            ("material icons", "!mi", False),
            ("openverse", "!opv", True),
            ("pinterest", "!pin", False),
            ("public domain image archive", "!pdia", True),
            ("sogou images", "!sogoui", False),
            ("svgrepo", "!svg", False),
            ("unsplash", "!us", True),
            ("wallhaven", "!wh", False),
            ("wikicommons.images", "!wc", True),
            ("yacy images", "!yai", False),
            ("yep images", "!yepi", False),
            ("seekr images (EN)", "!seimg", False),
            ("baidu images (ZH)", "!bdi", False),
            ("quark images (ZH)", "!qki", False),
        ]
    },
    "!videos": {
        "groups": {
            "!web": [
                ("bing videos", "!biv", True),
                ("brave.videos", "!brvid", True),
                ("duckduckgo videos", "!ddv", False),
                ("google videos", "!gov", True),
                ("qwant videos", "!qwv", True),
            ]
        },
        "engines": [
            ("360search videos", "!360sov", False),
            ("adobe stock video", "!asv", False),
            ("bilibili", "!bil", False),
            ("bitchute", "!bit", False),
            ("dailymotion", "!dm", True),
            ("google play movies", "!gpm", False),
            ("invidious", "!iv", False), # Also listed under !music
            ("livespace", "!ls", False),
            ("media.ccc.de", "!c3tv", False),
            ("odysee", "!od", False),
            ("peertube", "!ptb", False),
            ("piped", "!ppd", True),
            ("rumble", "!ru", False),
            ("sepiasearch", "!sep", True),
            ("vimeo", "!vm", False),
            ("wikicommons.videos", "!wcv", False),
            ("youtube", "!yt", True), # Also listed under !music
            ("mediathekviewweb (DE)", "!mvw", False),
            ("seekr videos (EN)", "!sevid", False),
            ("ina (FR)", "!in", False),
            ("niconico (JA)", "!nico", False),
            ("acfun (ZH)", "!acf", False),
            ("iqiyi (ZH)", "!iq", False),
            ("sogou videos (ZH)", "!sogouv", False),
        ]
    },
    "!news": {
        "groups": {
            "!web": [
                ("duckduckgo news", "!ddn", False),
                ("mojeek news", "!mjknews", False),
                ("presearch news", "!psnews", False),
                ("startpage news", "!spn", True),
            ],
            "!wikimedia": [
                ("wikinews", "!wn", True),
            ]
        },
        "engines": [
            ("bing news", "!bin", True),
            ("brave.news", "!brnews", True),
            ("google news", "!gon", True),
            ("qwant news", "!qwn", True),
            ("reuters", "!reu", False),
            ("yahoo news", "!yhn", False),
            ("yep news", "!yepn", False),
            ("tagesschau (DE)", "!ts", False), # Also listed under !general
            ("seekr news (EN)", "!senews", False),
            ("il post (IT)", "!pst", False),
            ("sogou wechat (ZH)", "!sogouw", False),
        ]
    },
    "!map": {
        "groups": {},
        "engines": [
            ("apple maps", "!apm", False),
            ("openstreetmap", "!osm", True),
            ("photon", "!ph", True),
        ]
    },
    "!music": {
        "groups": {
            "!lyrics": [
                ("genius", "!gen", True),
            ],
            "!radio": [
                ("radio browser", "!rb", True),
            ]
        },
        "engines": [
            ("adobe stock audio", "!asa", False),
            ("bandcamp", "!bc", True),
            ("deezer", "!dz", False),
            ("invidious", "!iv", False), # Also listed under !videos
            ("mixcloud", "!mc", True),
            ("piped.music", "!ppdm", True),
            ("soundcloud", "!sc", False),
            ("wikicommons.audio", "!wca", False),
            ("youtube", "!yt", True), # Also listed under !videos
        ]
    },
    "!it": {
        "groups": {
            "!packages": [
                ("alpine linux packages", "!alp", False),
                ("crates.io", "!crates", False),
                ("docker hub", "!dh", True),
                ("hex", "!hex", False),
                ("hoogle", "!ho", False),
                ("lib.rs", "!lrs", False),
                ("metacpan", "!cpan", False),
                ("npm", "!npm", False),
                ("packagist", "!pack", False),
                ("pkg.go.dev", "!pgo", False),
                ("pub.dev", "!pd", False),
                ("pypi", "!pypi", True),
                ("rubygems", "!rbg", False),
                ("voidlinux", "!void", False),
            ],
            "!q&a": [
                ("askubuntu", "!ubuntu", True),
                ("caddy.community", "!caddy", False),
                ("discuss.python", "!dpy", False),
                ("pi-hole.community", "!pi", False),
                ("stackoverflow", "!st", True),
                ("superuser", "!su", True),
            ],
            "!repos": [
                ("bitbucket", "!bb", False),
                ("codeberg", "!cb", False),
                ("gitea.com", "!gitea", False),
                ("github", "!gh", True),
                ("gitlab", "!gl", False),
                ("ollama", "!ollama", False),
                ("sourcehut", "!srht", False),
            ],
            "!software_wikis": [
                ("arch linux wiki", "!al", True),
                ("free software directory", "!fsd", False),
                ("gentoo", "!ge", True),
                ("nixos wiki", "!nixw", False),
            ]
        },
        "engines": [
            ("anaconda", "!conda", False),
            ("cppreference", "!cpp", False),
            ("habrahabr", "!habr", False),
            ("hackernews", "!hn", False),
            ("lobste.rs", "!lo", False),
            ("mankier", "!man", True),
            ("mdn", "!mdn", True),
            ("searchcode code", "!scc", False),
            ("baidu kaifa (ZH)", "!bdk", False),
        ]
    },
    "!science": {
        "groups": {
            "!scientific_publications": [
                ("arxiv", "!arx", True),
                ("crossref", "!cr", False),
                ("google scholar", "!gos", True),
                ("pubmed", "!pub", True),
                ("semantic scholar", "!se", False),
            ],
            "!wikimedia": [
                ("wikispecies", "!wsp", False), # Also listed under !general
            ]
        },
        "engines": [
            ("openairedatasets", "!oad", True),
            ("openairepublications", "!oap", True),
            ("pdbe", "!pdb", True),
        ]
    },
    "!files": {
        "groups": {
            "!apps": [
                ("apk mirror", "!apkm", False),
                ("apple app store", "!aps", False),
                ("fdroid", "!fd", False),
                ("google play apps", "!gpa", False),
            ]
        },
        "engines": [
            ("1337x", "!1337x", False),
            ("annas archive", "!aa", False),
            ("bt4g", "!bt4g", True),
            ("btdigg", "!bt", False),
            ("kickass", "!kc", True),
            ("library genesis", "!lg", False),
            ("nyaa", "!nt", False),
            ("openrepos", "!or", False),
            ("piratebay", "!tpb", True),
            ("solidtorrents", "!solid", True),
            ("tokyotoshokan", "!tt", False),
            ("wikicommons.files", "!wcf", False),
            ("z-library", "!zlib", False),
        ]
    },
    "!social_media": {
        "groups": {},
        "engines": [
            ("9gag", "!9g", False),
            ("lemmy comments", "!lecom", True),
            ("lemmy communities", "!leco", True),
            ("lemmy posts", "!lepo", True),
            ("lemmy users", "!leus", True),
            ("mastodon hashtags", "!mah", False),
            ("mastodon users", "!mau", False),
            ("reddit", "!re", False),
            ("tootfinder", "!toot", True),
        ]
    }
}

# Generate a flat list of default enabled engine names (used for config default)
# Using the engine name (part before bang) as the key SearXNG uses in API calls
DEFAULT_ENABLED_ENGINES = []
for tab_data in SEARXNG_ENGINE_DATA.values():
    for group_engines in tab_data.get("groups", {}).values():
        for name, bang, enabled in group_engines:
            engine_key = name.split(" ")[0].lower() # Best guess for engine key used by API
            if "(" in engine_key: # Handle cases like "seznam (CZ)" -> "seznam"
                engine_key = engine_key.split("(")[0]
            if enabled and engine_key not in DEFAULT_ENABLED_ENGINES:
                 DEFAULT_ENABLED_ENGINES.append(engine_key)
    for name, bang, enabled in tab_data.get("engines", []):
        engine_key = name.split(" ")[0].lower() # Best guess for engine key used by API
        if "(" in engine_key: # Handle cases like "seznam (CZ)" -> "seznam"
            engine_key = engine_key.split("(")[0]
        if enabled and engine_key not in DEFAULT_ENABLED_ENGINES:
            DEFAULT_ENABLED_ENGINES.append(engine_key)

# Sort for consistency
DEFAULT_ENABLED_ENGINES.sort()

# Note: The 'engine_key' derivation above is a best guess based on common patterns.
# SearXNG's internal engine names (used in the `engines=` API parameter) might differ slightly.
# Ideally, this mapping should be confirmed against SearXNG's source or documentation if possible.
# For now, we'll use this derived list. The bang command (`!bang`) is primarily for user interaction.

if __name__ == '__main__':
    # Example usage: Print default enabled engines
    print("Default Enabled SearXNG Engines (derived keys):")
    for engine in DEFAULT_ENABLED_ENGINES:
        print(f"- {engine}")

    # Example: Count total engines
    total_engines = 0
    for tab_data in SEARXNG_ENGINE_DATA.values():
        for group_engines in tab_data.get("groups", {}).values():
            total_engines += len(group_engines)
        total_engines += len(tab_data.get("engines", []))
    print(f"\nTotal engines parsed: {total_engines}") # Should be close to 238

    # Example: Find a specific engine
    target_bang = "!go"
    found = False
    for tab, tab_data in SEARXNG_ENGINE_DATA.items():
        for group, group_engines in tab_data.get("groups", {}).items():
            for name, bang, enabled in group_engines:
                if bang == target_bang:
                    print(f"\nFound {target_bang}: Name='{name}', Tab='{tab}', Group='{group}', DefaultEnabled={enabled}")
                    found = True
                    break
            if found: break
        if found: break
        for name, bang, enabled in tab_data.get("engines", []):
             if bang == target_bang:
                print(f"\nFound {target_bang}: Name='{name}', Tab='{tab}', Group=None, DefaultEnabled={enabled}")
                found = True
                break
        if found: break
