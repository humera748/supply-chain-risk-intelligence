"""
src/ports.py - Approximate Proxy Port & Maritime Lane Mapping
==============================================================
Maps 'Order Country' and 'Order Region' from the DataCo Supply Chain dataset
to major global commercial container seaports.

IMPORTANT DISCLAIMER:
DataCo records represent global commercial orders. Because actual bills of
lading and maritime port terminals are not directly recorded in DataCo,
this module implements an APPROXIMATE PROXY MAPPING to represent international
long-haul maritime container corridors (Origin Port -> Destination Port).
"""

from typing import Tuple
import pandas as pd


# ---------------------------------------------------------------------------
# 1. ORIGIN PORTS MAPPING (US & Puerto Rico Fulfillment Centroids)
# ---------------------------------------------------------------------------
# Origin shipments originate either from Puerto Rico (island maritime hub)
# or mainland US coastal maritime gateways based on state clusters.
STATE_TO_ORIGIN_PORT = {
    # West Coast Gateway (Trans-Pacific & West Coast trades)
    "CA": "Port of Los Angeles",
    "California": "Port of Los Angeles",
    "WA": "Port of Seattle",
    "Washington": "Port of Seattle",
    "OR": "Port of Los Angeles",
    "Oregon": "Port of Los Angeles",
    "NV": "Port of Los Angeles",
    "Nevada": "Port of Los Angeles",
    "AZ": "Port of Los Angeles",
    "Arizona": "Port of Los Angeles",
    
    # East Coast / North Atlantic Gateway (Trans-Atlantic trades)
    "NY": "Port of New York & New Jersey",
    "New York": "Port of New York & New Jersey",
    "NJ": "Port of New York & New Jersey",
    "New Jersey": "Port of New York & New Jersey",
    "PA": "Port of New York & New Jersey",
    "Pennsylvania": "Port of New York & New Jersey",
    "MA": "Port of Boston",
    "Massachusetts": "Port of Boston",
    "MD": "Port of Baltimore",
    "Maryland": "Port of Baltimore",
    "VA": "Port of Virginia (Norfolk)",
    "Virginia": "Port of Virginia (Norfolk)",
    
    # Southeast & Gulf Coast Gateways (LATAM & Trans-Atlantic)
    "FL": "Port of Miami",
    "Florida": "Port of Miami",
    "GA": "Port of Savannah",
    "Georgia": "Port of Savannah",
    "NC": "Port of Savannah",
    "North Carolina": "Port of Savannah",
    "SC": "Port of Charleston",
    "South Carolina": "Port of Charleston",
    "TX": "Port of Houston",
    "Texas": "Port of Houston",
    "LA": "Port of New Orleans",
    "Louisiana": "Port of New Orleans",
    
    # Caribbean Island Hub
    "PR": "Port of San Juan",
    "Puerto Rico": "Port of San Juan",
}

DEFAULT_US_ORIGIN_PORT = "Port of Savannah"
PR_ORIGIN_PORT = "Port of San Juan"


# ---------------------------------------------------------------------------
# 2. DESTINATION PORTS LOOKUP TABLE (Country-Level Primary Mapping)
# ---------------------------------------------------------------------------
# Maps destination country to its premier container port terminal.
COUNTRY_TO_DEST_PORT = {
    # Western & Central Europe
    "Francia": "Port of Le Havre",
    "France": "Port of Le Havre",
    "Alemania": "Port of Hamburg",
    "Germany": "Port of Hamburg",
    "Reino Unido": "Port of Felixstowe",
    "United Kingdom": "Port of Felixstowe",
    "Países Bajos": "Port of Rotterdam",
    "Netherlands": "Port of Rotterdam",
    "Bélgica": "Port of Antwerp",
    "Belgium": "Port of Antwerp",
    "Irlanda": "Port of Dublin",
    "Ireland": "Port of Dublin",
    "Suiza": "Port of Rotterdam",  # Inland via Rhine/Rotterdam corridor
    "Austria": "Port of Hamburg",
    
    # Southern Europe & Mediterranean
    "Italia": "Port of Genoa",
    "Italy": "Port of Genoa",
    "España": "Port of Valencia",
    "Spain": "Port of Valencia",
    "Portugal": "Port of Sines",
    "Grecia": "Port of Piraeus",
    "Greece": "Port of Piraeus",
    "Turquía": "Port of Ambarli (Istanbul)",
    "Turkey": "Port of Ambarli (Istanbul)",
    
    # Northern & Eastern Europe
    "Suecia": "Port of Gothenburg",
    "Sweden": "Port of Gothenburg",
    "Noruega": "Port of Oslo",
    "Norway": "Port of Oslo",
    "Finlandia": "Port of Helsinki",
    "Dinamarca": "Port of Aarhus",
    "Polonia": "Port of Gdansk",
    "Poland": "Port of Gdansk",
    "Rusia": "Port of St. Petersburg",
    "Russia": "Port of St. Petersburg",
    
    # East Asia & Pacific
    "China": "Port of Shanghai",
    "Japón": "Port of Yokohama",
    "Japan": "Port of Yokohama",
    "Corea del Sur": "Port of Busan",
    "South Korea": "Port of Busan",
    "Taiwán": "Port of Kaohsiung",
    "Taiwan": "Port of Kaohsiung",
    "Hong Kong": "Port of Hong Kong",
    "Australia": "Port of Sydney",
    "Nueva Zelanda": "Port of Auckland",
    "New Zealand": "Port of Auckland",
    
    # Southeast Asia
    "Singapur": "Port of Singapore",
    "Singapore": "Port of Singapore",
    "Indonesia": "Port of Tanjung Priok (Jakarta)",
    "Malasia": "Port of Port Klang",
    "Malaysia": "Port of Port Klang",
    "Tailandia": "Port of Laem Chabang",
    "Thailand": "Port of Laem Chabang",
    "Vietnam": "Port of Cai Mep",
    "Filipinas": "Port of Manila",
    "Philippines": "Port of Manila",
    
    # South Asia
    "India": "Port of Jawaharlal Nehru (Nhava Sheva/Mumbai)",
    "Pakistán": "Port of Karachi",
    "Pakistan": "Port of Karachi",
    "Bangladés": "Port of Chittagong",
    "Bangladesh": "Port of Chittagong",
    "Sri Lanka": "Port of Colombo",
    
    # Middle East / West Asia
    "Emiratos Árabes Unidos": "Port of Jebel Ali (Dubai)",
    "United Arab Emirates": "Port of Jebel Ali (Dubai)",
    "Arabia Saudita": "Port of King Abdulaziz (Dammam)",
    "Saudi Arabia": "Port of King Abdulaziz (Dammam)",
    "Israel": "Port of Haifa",
    "Jordania": "Port of Aqaba",
    
    # Latin America (LATAM) & Caribbean
    "México": "Port of Manzanillo",
    "Mexico": "Port of Manzanillo",
    "Brasil": "Port of Santos",
    "Brazil": "Port of Santos",
    "Colombia": "Port of Cartagena",
    "Chile": "Port of San Antonio (Valparaiso)",
    "Argentina": "Port of Buenos Aires",
    "Perú": "Port of Callao",
    "Peru": "Port of Callao",
    "Ecuador": "Port of Guayaquil",
    "Panamá": "Port of Balboa",
    "Panama": "Port of Balboa",
    "Costa Rica": "Port of Limon / Moin",
    "Guatemala": "Port of Puerto Quetzal",
    "Honduras": "Port of Puerto Cortes",
    "El Salvador": "Port of Acajutla",
    "Nicaragua": "Port of Corinto",
    "República Dominicana": "Port of Caucedo",
    "Dominican Republic": "Port of Caucedo",
    "Cuba": "Port of Mariel (Havana)",
    "Puerto Rico": "Port of San Juan",
    "Jamaica": "Port of Kingston",
    "Trinidad y Tobago": "Port of Port of Spain",
    
    # Africa
    "Egipto": "Port Said",
    "Egypt": "Port Said",
    "Marruecos": "Port of Tanger Med",
    "Morocco": "Port of Tanger Med",
    "Sudáfrica": "Port of Durban",
    "South Africa": "Port of Durban",
    "Nigeria": "Port of Lagos (Apapa)",
    "Kenia": "Port of Mombasa",
    "Kenya": "Port of Mombasa",
    "Ghana": "Port of Tema",
    
    # North America
    "Canadá": "Port of Vancouver",
    "Canada": "Port of Vancouver",
    "Estados Unidos": "Port of New York & New Jersey",
}


# ---------------------------------------------------------------------------
# 3. REGION-LEVEL FALLBACK TABLE (Ensures 100% Deterministic Coverage)
# ---------------------------------------------------------------------------
REGION_TO_DEST_PORT = {
    "Western Europe": "Port of Rotterdam",
    "Northern Europe": "Port of Felixstowe",
    "Southern Europe": "Port of Valencia",
    "Eastern Europe": "Port of Gdansk",
    "Eastern Asia": "Port of Shanghai",
    "Southeast Asia": "Port of Singapore",
    "South Asia": "Port of Jawaharlal Nehru (Nhava Sheva/Mumbai)",
    "Central Asia": "Port of Aktau",
    "West Asia": "Port of Jebel Ali (Dubai)",
    "Oceania": "Port of Sydney",
    "Central America": "Port of Manzanillo",
    "South America": "Port of Santos",
    "Caribbean": "Port of Kingston",
    "North Africa": "Port of Tanger Med",
    "West Africa": "Port of Lagos (Apapa)",
    "East Africa": "Port of Mombasa",
    "Central Africa": "Port of Douala",
    "Southern Africa": "Port of Durban",
    "Canada": "Port of Vancouver",
    "US Center": "Port of Houston",
    "West of USA": "Port of Los Angeles",
    "East of USA": "Port of New York & New Jersey",
    "South of  USA": "Port of Savannah",
}


def get_origin_port(customer_country: str, customer_state: str) -> str:
    """
    Determine approximate proxy origin seaport from customer origin location.
    """
    c_country = str(customer_country).strip()
    c_state = str(customer_state).strip()

    if c_country == "Puerto Rico" or c_state in ("PR", "Puerto Rico"):
        return PR_ORIGIN_PORT

    return STATE_TO_ORIGIN_PORT.get(c_state, DEFAULT_US_ORIGIN_PORT)


def get_dest_port(order_country: str, order_region: str) -> str:
    """
    Determine approximate proxy destination seaport from order country/region.
    """
    c_country = str(order_country).strip()
    c_region = str(order_region).strip()

    # 1. Direct country match
    if c_country in COUNTRY_TO_DEST_PORT:
        return COUNTRY_TO_DEST_PORT[c_country]

    # 2. Case-insensitive / normalized match
    for k, v in COUNTRY_TO_DEST_PORT.items():
        if k.lower() == c_country.lower():
            return v

    # 3. Regional gateway fallback
    if c_region in REGION_TO_DEST_PORT:
        return REGION_TO_DEST_PORT[c_region]

    # Clean whitespace in region string (e.g., 'South of  USA ')
    c_region_clean = " ".join(c_region.split())
    for r_k, r_v in REGION_TO_DEST_PORT.items():
        if " ".join(r_k.split()).lower() == c_region_clean.lower():
            return r_v

    # Global container crossroad default fallback
    return "Port of Rotterdam"


def assign_ports_and_lanes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds 'origin_port', 'dest_port', and 'lane' columns to the dataframe.
    """
    df = df.copy()
    
    # Vectorized / apply port assignment
    origin_ports = [
        get_origin_port(c_c, c_s)
        for c_c, c_s in zip(df["Customer Country"], df["Customer State"])
    ]
    
    dest_ports = [
        get_dest_port(o_c, o_r)
        for o_c, o_r in zip(df["Order Country"], df["Order Region"])
    ]
    
    df["origin_port"] = origin_ports
    df["dest_port"] = dest_ports
    df["lane"] = [f"{o} -> {d}" for o, d in zip(origin_ports, dest_ports)]
    return df


if __name__ == "__main__":
    # Test suite and summary
    sample_data = pd.DataFrame({
        "Customer Country": ["EE. UU.", "Puerto Rico", "EE. UU.", "EE. UU."],
        "Customer State": ["CA", "PR", "FL", "NY"],
        "Order Country": ["Francia", "Estados Unidos", "China", "India"],
        "Order Region": ["Western Europe", "East of USA", "Eastern Asia", "South Asia"]
    })
    mapped = assign_ports_and_lanes(sample_data)
    print("Sample Port & Lane Mappings (Proxy Approximation):")
    for _, row in mapped.iterrows():
        print(f"[{row['Customer State']}] -> [{row['Order Country']} ({row['Order Region']})]")
        print(f"  Lane: {row['lane']}")
