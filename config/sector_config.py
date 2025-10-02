"""
Sector-specific configuration for company analysis.

This module defines sector-specific metrics, factors, and companies
based on industry best practices and financial research.
"""

from typing import Dict, List

# Semiconductor sector companies (50+ companies for robust ML training)
SEMICONDUCTOR_COMPANIES = {
    # Tier 1: Mega Cap (>$100B)
    'NVDA': {'name': 'NVIDIA Corporation', 'segment': 'GPU/AI Chips'},
    'AVGO': {'name': 'Broadcom Inc.', 'segment': 'Diversified'},
    'TSM': {'name': 'Taiwan Semiconductor', 'segment': 'Pure Foundry'},

    # Tier 2: Large Cap ($20B-$100B)
    'AMD': {'name': 'Advanced Micro Devices', 'segment': 'CPU/GPU'},
    'MU': {'name': 'Micron Technology', 'segment': 'Memory/DRAM'},
    'QCOM': {'name': 'Qualcomm Inc.', 'segment': 'Mobile/RF'},
    'INTC': {'name': 'Intel Corporation', 'segment': 'CPU/Foundry'},
    'TXN': {'name': 'Texas Instruments', 'segment': 'Analog'},
    'ADI': {'name': 'Analog Devices', 'segment': 'Analog'},
    'MRVL': {'name': 'Marvell Technology', 'segment': 'Data Infrastructure'},
    'NXPI': {'name': 'NXP Semiconductors', 'segment': 'Automotive'},
    'ARM': {'name': 'Arm Holdings', 'segment': 'IP Licensing'},

    # Tier 3: Mid Cap ($5B-$20B)
    'MPWR': {'name': 'Monolithic Power Systems', 'segment': 'Power Management'},
    'MCHP': {'name': 'Microchip Technology', 'segment': 'Microcontrollers'},
    'ON': {'name': 'ON Semiconductor', 'segment': 'Power/Sensors'},
    'STM': {'name': 'STMicroelectronics', 'segment': 'Diversified'},
    'SWKS': {'name': 'Skyworks Solutions', 'segment': 'RF/Wireless'},
    'QRVO': {'name': 'Qorvo Inc.', 'segment': 'RF/Wireless'},
    'MXL': {'name': 'MaxLinear Inc.', 'segment': 'Connectivity'},
    'CRUS': {'name': 'Cirrus Logic', 'segment': 'Audio'},
    'SLAB': {'name': 'Silicon Labs', 'segment': 'IoT/Wireless'},
    'ALGM': {'name': 'Allegro MicroSystems', 'segment': 'Sensors'},
    'DIOD': {'name': 'Diodes Incorporated', 'segment': 'Discrete'},
    'SMTC': {'name': 'Semtech Corporation', 'segment': 'Analog/Mixed Signal'},
    'LITE': {'name': 'Lumentum Holdings', 'segment': 'Optical'},
    'COHR': {'name': 'Coherent Corp', 'segment': 'Optical/Lasers'},
    'WOLF': {'name': 'Wolfspeed Inc.', 'segment': 'SiC/Power'},

    # Tier 4: Small-Mid Cap ($1B-$5B)
    'AMBA': {'name': 'Ambarella Inc.', 'segment': 'Computer Vision'},
    'POWI': {'name': 'Power Integrations', 'segment': 'Power Management'},
    'MTSI': {'name': 'MACOM Technology', 'segment': 'RF/Microwave'},
    'SITM': {'name': 'SiTime Corporation', 'segment': 'MEMS Timing'},
    'CCMP': {'name': 'Cabot Microelectronics', 'segment': 'Materials'},
    'SMCI': {'name': 'Super Micro Computer', 'segment': 'Systems/AI Servers'},
    'RMBS': {'name': 'Rambus Inc.', 'segment': 'Memory IP'},
    'AOSL': {'name': 'Alpha and Omega', 'segment': 'Power Discretes'},
    'NVTS': {'name': 'Navitas Semiconductor', 'segment': 'GaN Power'},
    'PI': {'name': 'Impinj Inc.', 'segment': 'RFID/IoT'},
    'CRUS': {'name': 'Cirrus Logic', 'segment': 'Mixed Signal'},

    # Tier 5: Equipment (for diversification)
    'AMAT': {'name': 'Applied Materials', 'segment': 'Equipment'},
    'LRCX': {'name': 'Lam Research', 'segment': 'Equipment'},
    'KLAC': {'name': 'KLA Corporation', 'segment': 'Equipment'},
    'ASML': {'name': 'ASML Holding', 'segment': 'Lithography Equipment'},
    'ENTG': {'name': 'Entegris Inc.', 'segment': 'Materials/Equipment'},
    'MKSI': {'name': 'MKS Instruments', 'segment': 'Equipment'},
    'ACLS': {'name': 'Axcelis Technologies', 'segment': 'Ion Implant'},
    'UCTT': {'name': 'Ultra Clean Holdings', 'segment': 'Equipment/Services'},
    'FORM': {'name': 'FormFactor Inc.', 'segment': 'Test/Measurement'},
    'ONTO': {'name': 'Onto Innovation', 'segment': 'Process Control'},
    'COHU': {'name': 'Cohu Inc.', 'segment': 'Test/Handler Equipment'},
    'ICHR': {'name': 'Ichor Holdings', 'segment': 'Fluid Delivery'},
}

# Cloud/SaaS sector companies
CLOUD_SAAS_COMPANIES = {
    'CRM': {'name': 'Salesforce', 'segment': 'CRM Platform'},
    'NOW': {'name': 'ServiceNow', 'segment': 'IT Service Management'},
    'SNOW': {'name': 'Snowflake', 'segment': 'Data Cloud'},
    'DDOG': {'name': 'Datadog', 'segment': 'Monitoring'},
    'WDAY': {'name': 'Workday', 'segment': 'HR/Finance'},
    'ZS': {'name': 'Zscaler', 'segment': 'Security'},
    'CRWD': {'name': 'CrowdStrike', 'segment': 'Cybersecurity'},
    'NET': {'name': 'Cloudflare', 'segment': 'Edge/CDN'},
    'MDB': {'name': 'MongoDB', 'segment': 'Database'},
    'OKTA': {'name': 'Okta', 'segment': 'Identity'},
    'ZM': {'name': 'Zoom', 'segment': 'Communications'},
    'TEAM': {'name': 'Atlassian', 'segment': 'Collaboration'},
}

# Consumer Staples sector
CONSUMER_STAPLES_COMPANIES = {
    'KO': {'name': 'Coca-Cola', 'segment': 'Beverages'},
    'PEP': {'name': 'PepsiCo', 'segment': 'Food & Beverage'},
    'WMT': {'name': 'Walmart', 'segment': 'Retail'},
    'COST': {'name': 'Costco', 'segment': 'Warehouse Retail'},
    'PG': {'name': 'Procter & Gamble', 'segment': 'Consumer Goods'},
    'CL': {'name': 'Colgate-Palmolive', 'segment': 'Personal Care'},
    'MDLZ': {'name': 'Mondelez', 'segment': 'Snacks'},
    'GIS': {'name': 'General Mills', 'segment': 'Packaged Foods'},
    'K': {'name': 'Kellogg', 'segment': 'Cereals'},
    'KHC': {'name': 'Kraft Heinz', 'segment': 'Packaged Foods'},
    'CAG': {'name': 'Conagra Brands', 'segment': 'Packaged Foods'},
}


class SectorFactorDefinition:
    """
    Defines factor models for each sector based on industry research.

    Each factor is a composite of 3-5 related metrics that capture
    a fundamental dimension of company performance.

    Research sources:
    - Visible Alpha semiconductor KPIs
    - McKinsey semiconductor value creation
    - Industry-standard financial metrics
    """

    SEMICONDUCTOR_FACTORS = {
        'innovation_intensity': {
            'description': 'R&D spending, patent activity, and technological leadership',
            'weight': 0.20,
            'metrics': [
                'rd_pct_revenue',           # R&D as % of revenue (10-25% typical)
                'revenue_growth',           # YoY revenue growth
                'gross_margin',             # 30-70% depending on segment
            ],
            'benchmark': {
                'fabless': {'rd_pct_revenue': 0.15, 'gross_margin': 0.60},
                'foundry': {'rd_pct_revenue': 0.08, 'gross_margin': 0.45},
                'memory': {'rd_pct_revenue': 0.12, 'gross_margin': 0.35},
            }
        },

        'profitability_quality': {
            'description': 'Margin strength, capital efficiency, returns',
            'weight': 0.25,
            'metrics': [
                'operating_margin',         # Operating income / revenue
                'return_on_equity',         # Net income / equity
                'free_cash_flow_margin',    # FCF / revenue
            ],
            'benchmark': {
                'fabless': {'operating_margin': 0.25, 'roe': 0.20},
                'foundry': {'operating_margin': 0.35, 'roe': 0.15},
                'idm': {'operating_margin': 0.20, 'roe': 0.12},
            }
        },

        'market_position': {
            'description': 'Competitive positioning and pricing power',
            'weight': 0.20,
            'metrics': [
                'market_cap',               # Size proxy
                'revenue_rank_in_sector',   # Relative position
                'price_momentum_3m',        # Market perception
            ],
            'benchmark': {
                'leaders': {'market_cap': 100e9},  # $100B+
                'challengers': {'market_cap': 20e9},
            }
        },

        'financial_health': {
            'description': 'Balance sheet strength and liquidity',
            'weight': 0.15,
            'metrics': [
                'debt_to_equity',           # Total debt / equity
                'current_ratio',            # Current assets / current liabilities
                'quick_ratio',              # (CA - Inventory) / CL
            ],
            'benchmark': {
                'healthy': {'debt_to_equity': 0.3, 'current_ratio': 2.0},
                'stressed': {'debt_to_equity': 1.0, 'current_ratio': 1.0},
            }
        },

        'growth_momentum': {
            'description': 'Revenue and earnings acceleration',
            'weight': 0.20,
            'metrics': [
                'revenue_growth',           # YoY revenue growth
                'earnings_growth',          # YoY EPS growth
                'returns_3m',               # Stock momentum
                'returns_6m',
            ],
            'benchmark': {
                'high_growth': {'revenue_growth': 0.20},
                'mature': {'revenue_growth': 0.05},
            }
        },
    }

    CLOUD_SAAS_FACTORS = {
        'growth_efficiency': {
            'description': 'Rule of 40 (growth + margin), magic number',
            'weight': 0.30,
            'metrics': [
                'revenue_growth',
                'free_cash_flow_margin',
                'rule_of_40',  # revenue_growth + fcf_margin (should be > 40%)
            ],
            'benchmark': {
                'excellent': {'rule_of_40': 0.60},
                'good': {'rule_of_40': 0.40},
                'poor': {'rule_of_40': 0.20},
            }
        },

        'unit_economics': {
            'description': 'Customer acquisition and retention efficiency',
            'weight': 0.25,
            'metrics': [
                'gross_margin',             # SaaS should be 70%+
                'operating_margin',
                'fcf_margin',
            ],
            'benchmark': {
                'best_in_class': {'gross_margin': 0.75},
                'average': {'gross_margin': 0.65},
            }
        },

        'market_leadership': {
            'description': 'Market share and competitive moat',
            'weight': 0.20,
            'metrics': [
                'market_cap',
                'revenue_rank_in_sector',
                'price_momentum_6m',
            ]
        },

        'profitability_path': {
            'description': 'Path to profitability and cash generation',
            'weight': 0.25,
            'metrics': [
                'operating_margin',
                'fcf_margin',
                'return_on_equity',
            ]
        },
    }

    CONSUMER_STAPLES_FACTORS = {
        'brand_strength': {
            'description': 'Pricing power and market share',
            'weight': 0.25,
            'metrics': [
                'gross_margin',
                'operating_margin',
                'market_cap',
            ],
            'benchmark': {
                'premium': {'gross_margin': 0.55},
                'value': {'gross_margin': 0.25},
            }
        },

        'operational_excellence': {
            'description': 'Inventory and working capital efficiency',
            'weight': 0.20,
            'metrics': [
                'inventory_turnover',
                'return_on_assets',
                'asset_turnover',
            ]
        },

        'financial_stability': {
            'description': 'Defensive qualities and dividend capability',
            'weight': 0.25,
            'metrics': [
                'debt_to_equity',
                'interest_coverage',
                'free_cash_flow_margin',
            ]
        },

        'growth_resilience': {
            'description': 'Consistent growth in defensive sector',
            'weight': 0.30,
            'metrics': [
                'revenue_growth',
                'earnings_growth',
                'dividend_yield',
            ]
        },
    }

    @classmethod
    def get_factors(cls, sector: str) -> Dict:
        """Get factor definitions for a sector."""
        sector_map = {
            'semiconductors': cls.SEMICONDUCTOR_FACTORS,
            'cloud_saas': cls.CLOUD_SAAS_FACTORS,
            'consumer_staples': cls.CONSUMER_STAPLES_FACTORS,
        }
        return sector_map.get(sector, {})

    @classmethod
    def get_companies(cls, sector: str) -> Dict:
        """Get company list for a sector."""
        sector_map = {
            'semiconductors': SEMICONDUCTOR_COMPANIES,
            'cloud_saas': CLOUD_SAAS_COMPANIES,
            'consumer_staples': CONSUMER_STAPLES_COMPANIES,
        }
        return sector_map.get(sector, {})

    @classmethod
    def get_sector_etf(cls, sector: str) -> str:
        """Get benchmark ETF ticker for sector-relative calculations."""
        etf_map = {
            'semiconductors': 'SOXX',  # iShares Semiconductor ETF
            'cloud_saas': 'SKYY',      # First Trust Cloud Computing ETF
            'consumer_staples': 'XLP',  # Consumer Staples Select Sector SPDR
        }
        return etf_map.get(sector, 'SPY')  # Default to S&P 500


# Sector-specific target variable thresholds
SECTOR_TARGET_CONFIG = {
    'semiconductors': {
        'method': 'sector_relative',  # relative to SOXX ETF
        'lookback_period': '6M',       # 6-month returns
        'threshold_type': 'median',    # above/below sector median
        'volatility_adjust': True,     # Sharpe-like adjustment
    },
    'cloud_saas': {
        'method': 'sector_relative',
        'lookback_period': '6M',
        'threshold_type': 'median',
        'volatility_adjust': True,
    },
    'consumer_staples': {
        'method': 'sector_relative',
        'lookback_period': '12M',      # Longer for defensive sector
        'threshold_type': 'median',
        'volatility_adjust': True,
    },
}


def get_sector_info(sector: str) -> Dict:
    """
    Get all configuration for a specific sector.

    Args:
        sector: Sector name ('semiconductors', 'cloud_saas', 'consumer_staples')

    Returns:
        Dict containing companies, factors, benchmark ETF, and target config
    """
    return {
        'companies': SectorFactorDefinition.get_companies(sector),
        'factors': SectorFactorDefinition.get_factors(sector),
        'benchmark_etf': SectorFactorDefinition.get_sector_etf(sector),
        'target_config': SECTOR_TARGET_CONFIG.get(sector, {}),
        'name': sector.replace('_', ' ').title(),
    }
