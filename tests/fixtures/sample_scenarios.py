"""
Realistic test scenarios based on real decision-making contexts.
"""

# Scenario 1: Pricing Decision
PRICING_DECISION = {
    "name": "SaaS Price Increase",
    "dag": {
        "nodes": ["Price", "Brand", "Revenue", "CustomerAcquisition", "Churn"],
        "edges": [
            ["Price", "Revenue"],
            ["Price", "Churn"],
            ["Brand", "Price"],
            ["Brand", "CustomerAcquisition"],
            ["Brand", "Churn"],
            ["CustomerAcquisition", "Revenue"],
            ["Churn", "Revenue"],
        ],
    },
    "treatment": "Price",
    "outcome": "Revenue",
    "expected_adjustment_set": ["Brand"],
    "description": "Product team deciding whether to increase subscription price",
}

# Scenario 2: Feature Prioritization
FEATURE_PRIORITIZATION = {
    "name": "Onboarding vs Core Feature",
    "perspectives": [
        {
            "role": "Product Manager",
            "priorities": ["User acquisition", "Revenue", "Market share"],
            "constraints": ["Q4 deadline", "Limited budget"],
        },
        {
            "role": "Designer",
            "priorities": ["User experience", "Brand consistency"],
            "constraints": ["Design system limitations"],
        },
        {
            "role": "Engineer",
            "priorities": ["Code quality", "Maintainability"],
            "constraints": ["Team capacity", "Tech debt"],
        },
    ],
    "options": [
        {"id": "onboarding", "name": "Improve onboarding flow", "attributes": {}},
        {"id": "core_feature", "name": "Add power user feature", "attributes": {}},
        {"id": "refactor", "name": "Technical refactor", "attributes": {}},
    ],
    "expected_common_ground": ["User satisfaction", "Meet deadline"],
    "description": "Team aligning on Q4 roadmap priorities",
}

# Scenario 3: Marketing Investment
MARKETING_INVESTMENT = {
    "name": "Marketing Channel Allocation",
    "structural_model": {
        "variables": ["PaidAds", "ContentMarketing", "BrandAwareness", "Acquisition", "Revenue"],
        "equations": {
            "BrandAwareness": "baseline + 0.3*PaidAds + 0.5*ContentMarketing",
            "Acquisition": "1000 + 50*PaidAds + 30*BrandAwareness",
            "Revenue": "Acquisition * 50 - PaidAds*100 - ContentMarketing*80",
        },
        "distributions": {
            "baseline": {"type": "normal", "parameters": {"mean": 100, "std": 10}}
        },
    },
    "intervention": {"PaidAds": 20, "ContentMarketing": 10},
    "outcome": "Revenue",
    "expected_range": [45000, 65000],
    "description": "Marketing team allocating budget across channels",
}
