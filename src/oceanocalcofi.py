import base64
from pathlib import Path
from urllib.parse import quote

import joblib
import pandas as pd
import pydeck as pdk
import streamlit as st


# =========================================================
# CONFIGURACIÓN GENERAL
# =========================================================

st.set_page_config(
    page_title="Predictor de Especies Marinas",
    page_icon="🐟",
    layout="wide"
)


# =========================================================
# RUTAS DEL PROYECTO
# =========================================================

BASE_DIR = Path(__file__).resolve().parents[1]

RUTA_MODELO = (
    BASE_DIR
    / "notebooks"
    / "final"
    / "modelo_xgboost.pkl"
)

RUTA_DATASET = (
    BASE_DIR
    / "data"
    / "processed"
    / "dataset_final.csv"
)

RUTA_IMAGEN = (
    BASE_DIR
    / "src"
    / "assets"
    / "oceano.jpg"
)


# =========================================================
# VARIABLES DEL MODELO
# =========================================================

COLUMNAS_MODELO = [
    "Year",
    "Month",
    "Lat_Dec",
    "Lon_Dec",
    "T_degC",
    "Salnty",
    "STheta",
    "O2ml_L",
    "O2Sat",
    "Bottom_D",
    "Distance",
    "ChlorA",
    "Phaeop",
    "PO4uM",
    "SiO3uM",
    "NO2uM",
    "NO3uM",
    "IntChl",
    "Wind_Spd",
    "Wave_Ht"
]


MESES = {
    "Enero": 1,
    "Febrero": 2,
    "Marzo": 3,
    "Abril": 4,
    "Mayo": 5,
    "Junio": 6,
    "Julio": 7,
    "Agosto": 8,
    "Septiembre": 9,
    "Octubre": 10,
    "Noviembre": 11,
    "Diciembre": 12
}


# =========================================================
# CARGAR FOTO
# =========================================================

def imagen_a_base64(ruta):

    with open(ruta, "rb") as archivo:

        return base64.b64encode(
            archivo.read()
        ).decode()


if RUTA_IMAGEN.exists():

    imagen_hero = imagen_a_base64(
        RUTA_IMAGEN
    )

else:

    imagen_hero = ""


# =========================================================
# CARGAR MODELO
# =========================================================

@st.cache_resource
def cargar_modelo():

    return joblib.load(
        RUTA_MODELO
    )


modelo = cargar_modelo()


# =========================================================
# CARGAR DATASET
# =========================================================

@st.cache_data
def cargar_dataset():

    return pd.read_csv(
        RUTA_DATASET
    )


dataset = cargar_dataset()


# =========================================================
# FILTRAR SOLO LAS DOS ESPECIES
# =========================================================

ESPECIES_OBJETIVO = [
    "Engraulis mordax",
    "Sardinops sagax"
]


dataset_especies = (
    dataset[
        dataset["scientificName"].isin(
            ESPECIES_OBJETIVO
        )
    ]
    .copy()
)


# =========================================================
# PUNTOS REALES DE MUESTREO
# =========================================================

@st.cache_data
def cargar_puntos_reales():

    puntos = (
        dataset_especies[
            [
                "Lat_Dec",
                "Lon_Dec"
            ]
        ]
        .dropna()
        .drop_duplicates()
        .reset_index(
            drop=True
        )
    )

    return puntos


puntos_reales = cargar_puntos_reales()


# =========================================================
# RANGOS REALISTAS DEL DATASET
# =========================================================

def obtener_rango_realista(columna):

    serie = (
        dataset_especies[
            columna
        ]
        .dropna()
    )

    minimo = float(
        serie.quantile(
            0.01
        )
    )

    maximo = float(
        serie.quantile(
            0.99
        )
    )

    mediana = float(
        serie.median()
    )

    return {
        "min": minimo,
        "max": maximo,
        "mediana": mediana
    }


RANGOS = {}

for columna in COLUMNAS_MODELO:

    RANGOS[
        columna
    ] = obtener_rango_realista(
        columna
    )


# =========================================================
# AÑO Y MES
# =========================================================

RANGOS["Year"] = {
    "min": float(
        dataset_especies[
            "Year"
        ].min()
    ),

    "max": float(
        dataset_especies[
            "Year"
        ].max()
    ),

    "mediana": float(
        dataset_especies[
            "Year"
        ].median()
    )
}


RANGOS["Month"] = {
    "min": 1.0,
    "max": 12.0,
    "mediana": float(
        dataset_especies[
            "Month"
        ].median()
    )
}


# =========================================================
# COORDENADAS
# =========================================================

RANGOS["Lat_Dec"] = {
    "min": float(
        puntos_reales[
            "Lat_Dec"
        ].min()
    ),

    "max": float(
        puntos_reales[
            "Lat_Dec"
        ].max()
    ),

    "mediana": float(
        puntos_reales[
            "Lat_Dec"
        ].median()
    )
}


RANGOS["Lon_Dec"] = {
    "min": float(
        puntos_reales[
            "Lon_Dec"
        ].min()
    ),

    "max": float(
        puntos_reales[
            "Lon_Dec"
        ].max()
    ),

    "mediana": float(
        puntos_reales[
            "Lon_Dec"
        ].median()
    )
}


# =========================================================
# VALORES INICIALES
# =========================================================

mes_inicial_numero = int(
    round(
        RANGOS[
            "Month"
        ][
            "mediana"
        ]
    )
)


mes_inicial_numero = max(
    1,
    min(
        12,
        mes_inicial_numero
    )
)


MES_INICIAL = list(
    MESES.keys()
)[
    mes_inicial_numero - 1
]


VALORES_INICIALES = {

    "Year":
        int(
            RANGOS[
                "Year"
            ][
                "mediana"
            ]
        ),

    "Month":
        MES_INICIAL,

    "Lat_Dec":
        float(
            RANGOS[
                "Lat_Dec"
            ][
                "mediana"
            ]
        ),

    "Lon_Dec":
        float(
            RANGOS[
                "Lon_Dec"
            ][
                "mediana"
            ]
        ),

    "T_degC":
        float(
            RANGOS[
                "T_degC"
            ][
                "mediana"
            ]
        ),

    "Salnty":
        float(
            RANGOS[
                "Salnty"
            ][
                "mediana"
            ]
        ),

    "STheta":
        float(
            RANGOS[
                "STheta"
            ][
                "mediana"
            ]
        ),

    "O2ml_L":
        float(
            RANGOS[
                "O2ml_L"
            ][
                "mediana"
            ]
        ),

    "O2Sat":
        float(
            RANGOS[
                "O2Sat"
            ][
                "mediana"
            ]
        ),

    "Bottom_D":
        float(
            RANGOS[
                "Bottom_D"
            ][
                "mediana"
            ]
        ),

    "Distance":
        float(
            RANGOS[
                "Distance"
            ][
                "mediana"
            ]
        ),

    "ChlorA":
        float(
            RANGOS[
                "ChlorA"
            ][
                "mediana"
            ]
        ),

    "Phaeop":
        float(
            RANGOS[
                "Phaeop"
            ][
                "mediana"
            ]
        ),

    "PO4uM":
        float(
            RANGOS[
                "PO4uM"
            ][
                "mediana"
            ]
        ),

    "SiO3uM":
        float(
            RANGOS[
                "SiO3uM"
            ][
                "mediana"
            ]
        ),

    "NO2uM":
        float(
            RANGOS[
                "NO2uM"
            ][
                "mediana"
            ]
        ),

    "NO3uM":
        float(
            RANGOS[
                "NO3uM"
            ][
                "mediana"
            ]
        ),

    "IntChl":
        float(
            RANGOS[
                "IntChl"
            ][
                "mediana"
            ]
        ),

    "Wind_Spd":
        float(
            RANGOS[
                "Wind_Spd"
            ][
                "mediana"
            ]
        ),

    "Wave_Ht":
        float(
            RANGOS[
                "Wave_Ht"
            ][
                "mediana"
            ]
        )
}


# =========================================================
# VELERO
# =========================================================

svg_velero = """
<svg
xmlns="http://www.w3.org/2000/svg"
width="128"
height="128"
viewBox="0 0 128 128">

<circle
cx="64"
cy="64"
r="58"
fill="#ffffff"
fill-opacity="0.92"
/>

<line
x1="64"
y1="24"
x2="64"
y2="83"
stroke="#183d52"
stroke-width="5"
/>

<path
d="M60 28 L60 72 L30 72 Z"
fill="#00a6c8"
/>

<path
d="M68 34 L68 72 L94 72 Z"
fill="#f5a623"
/>

<path
d="M26 78 L102 78 L90 97 L40 97 Z"
fill="#183d52"
/>

<path
d="
M23 104
Q37 96 51 104
Q65 112 79 104
Q93 96 107 104
"
fill="none"
stroke="#2ba7d6"
stroke-width="5"
stroke-linecap="round"
/>

</svg>
"""


URL_VELERO = (
    "data:image/svg+xml;charset=utf-8,"
    +
    quote(
        svg_velero
    )
)


# =========================================================
# ESTILOS
# =========================================================

st.markdown(
    f"""
    <style>

    @import url(
    'https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Montserrat:wght@600;700;800&display=swap'
    );


    html,
    body,
    [class*="css"] {{
        font-family:
            'Inter',
            sans-serif;
    }}


    .stApp {{
        background:
            linear-gradient(
                180deg,
                #061923 0%,
                #0a2937 45%,
                #0d3443 100%
            );

        color: white;
    }}


    .block-container {{
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1400px;
    }}


    h1,
    h2,
    h3 {{
        font-family:
            'Montserrat',
            sans-serif !important;
    }}


    .hero {{
        background-image:

            linear-gradient(
                rgba(4,22,31,0.30),
                rgba(4,22,31,0.70)
            ),

            url(
                "data:image/jpeg;base64,{imagen_hero}"
            );


        background-size:
            cover;

        background-position:
            center;

        background-repeat:
            no-repeat;


        min-height:
            245px;


        display:
            flex;

        flex-direction:
            column;

        justify-content:
            flex-end;


        padding:
            2.2rem;


        border-radius:
            24px;


        border:
            1px solid
            rgba(
                255,
                255,
                255,
                0.15
            );


        box-shadow:
            0 12px 35px
            rgba(
                0,
                0,
                0,
                0.30
            );


        margin-bottom:
            1.5rem;
    }}


    .hero h1 {{
        font-family:
            'Montserrat',
            sans-serif;

        font-size:
            clamp(
                1.9rem,
                3vw,
                2.7rem
            );

        font-weight:
            800;

        color:
            white;

        margin:
            0 0 0.6rem 0;

        line-height:
            1.05;

        white-space:
            nowrap;
    }}


    .hero h2 {{
        font-size:
            1.15rem;

        font-weight:
            600;

        color:
            #d7f2ff;

        margin:
            0 0 0.8rem 0;
    }}


    .hero p {{
        font-size:
            0.98rem;

        line-height:
            1.6;

        max-width:
            900px;

        color:
            #edf9ff;

        margin:
            0;
    }}


    .escenario-grid {{
        display:
            grid;

        grid-template-columns:
            repeat(
                7,
                1fr
            );

        gap:
            10px;

        margin-top:
            0.8rem;

        margin-bottom:
            1.5rem;
    }}


    .escenario-card {{
        background:
            rgba(
                255,
                255,
                255,
                0.06
            );

        border:
            1px solid
            rgba(
                255,
                255,
                255,
                0.10
            );

        border-radius:
            14px;

        padding:
            0.9rem;

        text-align:
            center;
    }}


    .escenario-label {{
        color:
            #a8ccda;

        font-size:
            0.76rem;

        text-transform:
            uppercase;

        font-weight:
            700;

        margin-bottom:
            0.3rem;
    }}


    .escenario-valor {{
        color:
            white;

        font-weight:
            700;

        font-size:
            0.98rem;
    }}


    .resultado {{
        background:
            linear-gradient(
                135deg,
                rgba(
                    25,
                    90,
                    116,
                    0.30
                ),
                rgba(
                    18,
                    57,
                    73,
                    0.55
                )
            );

        border:
            1px solid
            rgba(
                255,
                255,
                255,
                0.12
            );

        border-radius:
            22px;

        padding:
            2rem;

        text-align:
            center;

        box-shadow:
            0 8px 25px
            rgba(
                0,
                0,
                0,
                0.20
            );

        margin-top:
            1rem;

        margin-bottom:
            1.5rem;
    }}


    .resultado h1 {{
        font-size:
            3rem;

        color:
            #8ce4ff;

        margin:
            0.2rem;
    }}


    .resultado h2 {{
        font-size:
            2rem;

        color:
            white;

        margin-bottom:
            0;
    }}


    div[data-testid="stButton"] button {{
        border-radius:
            12px;

        min-height:
            50px;

        font-family:
            'Montserrat',
            sans-serif;

        font-weight:
            700;
    }}


    div[data-testid="stDownloadButton"] button {{
        border-radius:
            12px;

        min-height:
            48px;

        font-weight:
            700;
    }}


    @media (
        max-width:
        1000px
    ) {{

        .escenario-grid {{
            grid-template-columns:
                repeat(
                    2,
                    1fr
                );
        }}


        .hero h1 {{
            white-space:
                normal;
        }}
    }}

    </style>
    """,

    unsafe_allow_html=True
)


# =========================================================
# SLIDERS
# =========================================================

def slider_controlado(
    etiqueta,
    columna,
    step,
    icono="",
    key=None
):

    minimo_real = (
        RANGOS[
            columna
        ][
            "min"
        ]
    )


    maximo_real = (
        RANGOS[
            columna
        ][
            "max"
        ]
    )


    texto = (
        f"{icono} {etiqueta}"
        if icono
        else
        etiqueta
    )


    return st.slider(

        texto,

        min_value=
            float(
                minimo_real
            ),

        max_value=
            float(
                maximo_real
            ),

        value=
            float(
                VALORES_INICIALES[
                    columna
                ]
            ),

        step=
            float(
                step
            ),

        key=key
    )


# =========================================================
# RESTABLECER TODO EL ESCENARIO
# =========================================================

def restablecer_escenario():

    for clave, valor in VALORES_INICIALES.items():

        st.session_state[
            clave
        ] = valor


# =========================================================
# CABECERA
# =========================================================

st.markdown(
    """
<div class="hero">

<h1>
🐟 PREDICTOR DE ESPECIES MARINAS
</h1>

<h2>
Basado en datos oceanográficos de CalCOFI
</h2>

<p>

Aplicación de Machine Learning para clasificar
condiciones oceanográficas y ambientales entre
<b>Anchoa</b>
(<i>Engraulis mordax</i>)
y
<b>Sardina</b>
(<i>Sardinops sagax</i>).

</p>

</div>
""",

    unsafe_allow_html=True
)


# =========================================================
# INFORMACIÓN DEL MODELO
# =========================================================

with st.expander(
    "ℹ️ Información del modelo"
):

    st.markdown(
        """
**Modelo:** XGBoost

**Fuente de datos:** CalCOFI

**Variables utilizadas:** 20

**Clases:**

- Anchoa — *Engraulis mordax*
- Sardina — *Sardinops sagax*

Los controles utilizan rangos representativos de los
registros de Anchoa y Sardina presentes en el dataset.

Para evitar que unos pocos valores extremos distorsionen
la interfaz, los sliders utilizan aproximadamente el
98 % central de las observaciones.
        """
    )


# =========================================================
# BOTÓN RESTABLECER
# =========================================================

st.button(
    "↺ Restablecer escenario",
    use_container_width=True,
    on_click=restablecer_escenario
)


# =========================================================
# LOCALIZACIÓN
# =========================================================

with st.container(
    border=True
):

    st.subheader(
        "📍 Localización del muestreo"
    )


    st.caption(
        "Latitud y longitud pueden moverse de forma independiente. "
        "La aplicación comprueba que existan muestras CalCOFI "
        "realmente próximas antes de permitir la predicción."
    )


    izquierda, derecha = st.columns(
        [
            0.85,
            1.5
        ]
    )


    with izquierda:


        Year = st.number_input(
            "Año",
            min_value=
                int(
                    RANGOS[
                        "Year"
                    ][
                        "min"
                    ]
                ),
            max_value=
                int(
                    RANGOS[
                        "Year"
                    ][
                        "max"
                    ]
                ),
            value=
                VALORES_INICIALES[
                    "Year"
                ],
            step=1,
            key="Year"
        )


        nombres_meses = list(
            MESES.keys()
        )


        Month_nombre = st.selectbox(
            "Mes",
            nombres_meses,
            index=nombres_meses.index(
                VALORES_INICIALES[
                    "Month"
                ]
            ),
            key="Month"
        )


        Month = MESES[
            Month_nombre
        ]


        Lat_Dec = st.slider(

            "Latitud",

            min_value=
                float(
                    RANGOS[
                        "Lat_Dec"
                    ][
                        "min"
                    ]
                ),

            max_value=
                float(
                    RANGOS[
                        "Lat_Dec"
                    ][
                        "max"
                    ]
                ),

            value=
                VALORES_INICIALES[
                    "Lat_Dec"
                ],

            step=0.1,

            key="Lat_Dec"
        )


        Lon_Dec = st.slider(

            "Longitud",

            min_value=
                float(
                    RANGOS[
                        "Lon_Dec"
                    ][
                        "min"
                    ]
                ),

            max_value=
                float(
                    RANGOS[
                        "Lon_Dec"
                    ][
                        "max"
                    ]
                ),

            value=
                VALORES_INICIALES[
                    "Lon_Dec"
                ],

            step=0.1,

            key="Lon_Dec"
        )


    # =====================================================
    # MAPA
    # =====================================================

    with derecha:


        punto_velero = pd.DataFrame(
            [
                {

                    "lat":
                        Lat_Dec,

                    "lon":
                        Lon_Dec,

                    "icon":
                        {

                            "url":
                                URL_VELERO,

                            "width":
                                128,

                            "height":
                                128,

                            "anchorY":
                                128
                        }
                }
            ]
        )


        capa_velero = pdk.Layer(

            "IconLayer",

            data=
                punto_velero,

            get_icon=
                "icon",

            get_position=[
                "lon",
                "lat"
            ],

            get_size=28,

            size_scale=1,

            pickable=True
        )


        vista_mapa = pdk.ViewState(

            latitude=
                Lat_Dec,

            longitude=
                Lon_Dec,

            zoom=5,

            pitch=0
        )


        mapa = pdk.Deck(

            map_provider=
                "carto",

            map_style=
                "light",

            initial_view_state=
                vista_mapa,

            layers=[
                capa_velero
            ],

            tooltip={
                "text":
                    "Coordenadas seleccionadas\n"
                    "Latitud: {lat}\n"
                    "Longitud: {lon}"
            }
        )


        st.pydeck_chart(
            mapa,
            use_container_width=True
        )


# =========================================================
# VALIDACIÓN GEOGRÁFICA
# =========================================================

distancias = (

    (
        puntos_reales[
            "Lat_Dec"
        ]
        -
        Lat_Dec
    ) ** 2

    +

    (
        puntos_reales[
            "Lon_Dec"
        ]
        -
        Lon_Dec
    ) ** 2
)


distancia_minima = float(
    distancias.min()
)


UMBRAL_MUESTREO = 0.08


coordenadas_validas = (
    distancia_minima
    <=
    UMBRAL_MUESTREO
)


if coordenadas_validas:

    st.success(
        "✅ Localización válida: "
        "existen muestras CalCOFI muy próximas a estas coordenadas."
    )

else:

    st.error(
        "❌ Localización fuera de la zona representada por las muestras. "
        "Mueve la latitud o la longitud hacia una zona de muestreo CalCOFI."
    )


# =========================================================
# CONDICIONES OCEANOGRÁFICAS
# =========================================================

with st.container(
    border=True
):

    st.subheader(
        "🌊 Condiciones oceanográficas"
    )


    st.caption(
        "Los controles utilizan los valores representativos "
        "observados para Anchoa y Sardina."
    )


    col1, col2 = st.columns(
        2
    )


    with col1:


        T_degC = slider_controlado(
            "Temperatura del agua (°C)",
            "T_degC",
            0.1,
            "🌡️",
            "T_degC"
        )


        Salnty = slider_controlado(
            "Salinidad",
            "Salnty",
            0.01,
            "🧂",
            "Salnty"
        )


        STheta = slider_controlado(
            "Densidad potencial",
            "STheta",
            0.1,
            "⚖️",
            "STheta"
        )


        O2ml_L = slider_controlado(
            "Oxígeno disuelto (ml/L)",
            "O2ml_L",
            0.1,
            "🫧",
            "O2ml_L"
        )


    with col2:


        O2Sat = slider_controlado(
            "Saturación de oxígeno (%)",
            "O2Sat",
            1.0,
            "💨",
            "O2Sat"
        )


        Bottom_D = slider_controlado(
            "Profundidad del fondo (m)",
            "Bottom_D",
            10.0,
            "🌊",
            "Bottom_D"
        )


        Distance = slider_controlado(
            "Distancia",
            "Distance",
            1.0,
            "📏",
            "Distance"
        )


# =========================================================
# NUTRIENTES Y BIOLOGÍA
# =========================================================

with st.container(
    border=True
):

    st.subheader(
        "🧪 Nutrientes y productividad biológica"
    )


    st.caption(
        "Los límites de las barras se calculan a partir "
        "de los registros de las dos especies."
    )


    col1, col2 = st.columns(
        2
    )


    with col1:


        ChlorA = slider_controlado(
            "Clorofila A",
            "ChlorA",
            0.1,
            "🌿",
            "ChlorA"
        )


        Phaeop = slider_controlado(
            "Feopigmentos",
            "Phaeop",
            0.05,
            "🧫",
            "Phaeop"
        )


        PO4uM = slider_controlado(
            "Fosfato (µM)",
            "PO4uM",
            0.1,
            "🧪",
            "PO4uM"
        )


        SiO3uM = slider_controlado(
            "Silicato (µM)",
            "SiO3uM",
            0.5,
            "🔬",
            "SiO3uM"
        )


    with col2:


        NO2uM = slider_controlado(
            "Nitrito (µM)",
            "NO2uM",
            0.01,
            "🧬",
            "NO2uM"
        )


        NO3uM = slider_controlado(
            "Nitrato (µM)",
            "NO3uM",
            0.5,
            "🧪",
            "NO3uM"
        )


        IntChl = slider_controlado(
            "Clorofila integrada",
            "IntChl",
            1.0,
            "🌱",
            "IntChl"
        )


# =========================================================
# CONDICIONES AMBIENTALES
# =========================================================

with st.container(
    border=True
):

    st.subheader(
        "🌬️ Condiciones ambientales"
    )


    st.caption(
        "Los valores disponibles se mantienen dentro "
        "del comportamiento representativo del dataset."
    )


    col1, col2 = st.columns(
        2
    )


    with col1:


        Wind_Spd = slider_controlado(
            "Velocidad del viento",
            "Wind_Spd",
            1.0,
            "🌬️",
            "Wind_Spd"
        )


    with col2:


        Wave_Ht = slider_controlado(
            "Altura de las olas (m)",
            "Wave_Ht",
            0.1,
            "〰️",
            "Wave_Ht"
        )


# =========================================================
# ESCENARIO ACTUAL HORIZONTAL
# =========================================================

st.subheader(
    "🧭 Escenario actual"
)


st.markdown(
    f"""
<div class="escenario-grid">


<div class="escenario-card">

<div class="escenario-label">
Fecha
</div>

<div class="escenario-valor">
📅 {Month_nombre} {Year}
</div>

</div>


<div class="escenario-card">

<div class="escenario-label">
Temperatura
</div>

<div class="escenario-valor">
🌡️ {T_degC:.1f} °C
</div>

</div>


<div class="escenario-card">

<div class="escenario-label">
Profundidad
</div>

<div class="escenario-valor">
🌊 {Bottom_D:.0f} m
</div>

</div>


<div class="escenario-card">

<div class="escenario-label">
Salinidad
</div>

<div class="escenario-valor">
🧂 {Salnty:.2f}
</div>

</div>


<div class="escenario-card">

<div class="escenario-label">
Viento
</div>

<div class="escenario-valor">
🌬️ {Wind_Spd:.1f}
</div>

</div>


<div class="escenario-card">

<div class="escenario-label">
Oleaje
</div>

<div class="escenario-valor">
〰️ {Wave_Ht:.1f} m
</div>

</div>


<div class="escenario-card">

<div class="escenario-label">
Coordenadas
</div>

<div class="escenario-valor">
📍 {Lat_Dec:.2f} / {Lon_Dec:.2f}
</div>

</div>


</div>
""",

    unsafe_allow_html=True
)


# =========================================================
# ESTADO DEL ESCENARIO
# =========================================================

st.subheader(
    "🛡️ Estado del escenario"
)


if coordenadas_validas:

    st.success(
        "✅ Escenario válido. "
        "La localización está representada por muestras muy próximas "
        "y las variables están dentro de rangos representativos "
        "de Anchoa y Sardina."
    )

else:

    st.error(
        "❌ La localización no está suficientemente próxima "
        "a las zonas representadas por las muestras."
    )


# =========================================================
# ANALIZAR
# =========================================================

if st.button(
    "🔎 ANALIZAR ESCENARIO",
    use_container_width=True
):


    if not coordenadas_validas:

        st.error(
            "No se puede realizar la predicción: "
            "la localización está demasiado alejada "
            "de las zonas representadas por las muestras."
        )

        st.stop()


    datos = pd.DataFrame(
        [
            {

                "Year":
                    Year,

                "Month":
                    Month,

                "Lat_Dec":
                    Lat_Dec,

                "Lon_Dec":
                    Lon_Dec,

                "T_degC":
                    T_degC,

                "Salnty":
                    Salnty,

                "STheta":
                    STheta,

                "O2ml_L":
                    O2ml_L,

                "O2Sat":
                    O2Sat,

                "Bottom_D":
                    Bottom_D,

                "Distance":
                    Distance,

                "ChlorA":
                    ChlorA,

                "Phaeop":
                    Phaeop,

                "PO4uM":
                    PO4uM,

                "SiO3uM":
                    SiO3uM,

                "NO2uM":
                    NO2uM,

                "NO3uM":
                    NO3uM,

                "IntChl":
                    IntChl,

                "Wind_Spd":
                    Wind_Spd,

                "Wave_Ht":
                    Wave_Ht
            }
        ]
    )


    prediccion = modelo.predict(
        datos
    )[0]


    probabilidades = modelo.predict_proba(
        datos
    )[0]


    prob_anchoa = float(
        probabilidades[
            0
        ]
    )


    prob_sardina = float(
        probabilidades[
            1
        ]
    )


    if prediccion == 1:

        especie = "Sardina"

        nombre_cientifico = (
            "Sardinops sagax"
        )

        probabilidad_principal = (
            prob_sardina
        )


    else:

        especie = "Anchoa"

        nombre_cientifico = (
            "Engraulis mordax"
        )

        probabilidad_principal = (
            prob_anchoa
        )


    diferencia = abs(
        prob_anchoa
        -
        prob_sardina
    )


    if diferencia < 0.10:

        nivel_resultado = (
            "Resultado muy ajustado"
        )


    elif diferencia < 0.30:

        nivel_resultado = (
            "Resultado moderadamente definido"
        )


    else:

        nivel_resultado = (
            "Resultado claramente definido"
        )


    st.subheader(
        "🎯 Resultado"
    )


    st.markdown(
        f"""
<div class="resultado">

<h2>
{especie}
</h2>

<p>
<i>
{nombre_cientifico}
</i>
</p>

<h1>
{probabilidad_principal:.1%}
</h1>

<p>
Probabilidad estimada por el modelo
para esta clase
</p>

<p>
<b>
{nivel_resultado}
</b>
</p>

</div>
""",

        unsafe_allow_html=True
    )


    st.subheader(
        "📊 Comparación entre especies"
    )


    st.write(
        f"### 🐟 Anchoa — {prob_anchoa:.1%}"
    )


    st.progress(
        int(
            prob_anchoa
            *
            100
        )
    )


    st.write(
        f"### 🐟 Sardina — {prob_sardina:.1%}"
    )


    st.progress(
        int(
            prob_sardina
            *
            100
        )
    )


    st.subheader(
        "🧠 Lectura del resultado"
    )


    if diferencia < 0.10:

        st.warning(
            "Resultado muy ajustado. "
            "Las probabilidades de las dos clases están muy próximas."
        )


    elif diferencia < 0.30:

        st.info(
            "Resultado moderadamente definido. "
            "El modelo muestra una preferencia, "
            "pero las dos clases siguen relativamente próximas."
        )


    else:

        st.success(
            "Resultado claramente separado. "
            "El modelo diferencia de forma notable las dos clases "
            "en este escenario."
        )


    with st.expander(
        "📋 Ver datos utilizados por el modelo"
    ):

        st.dataframe(
            datos,
            use_container_width=True
        )


    resultado_csv = (
        datos.copy()
    )


    resultado_csv[
        "Prediccion"
    ] = especie


    resultado_csv[
        "Nombre_cientifico"
    ] = nombre_cientifico


    resultado_csv[
        "Probabilidad_Anchoa"
    ] = prob_anchoa


    resultado_csv[
        "Probabilidad_Sardina"
    ] = prob_sardina


    csv = resultado_csv.to_csv(
        index=False
    ).encode(
        "utf-8-sig"
    )


    st.download_button(

        "⬇️ Descargar análisis en CSV",

        data=csv,

        file_name=
            "analisis_especies_calcofi.csv",

        mime=
            "text/csv",

        use_container_width=True
    )


# =========================================================
# INFORMACIÓN FINAL
# =========================================================

st.divider()


with st.expander(
    "ℹ️ Sobre este proyecto"
):

    st.markdown(
        """
**Modelo utilizado:** XGBoost

**Fuente de datos:** CalCOFI

La aplicación utiliza exclusivamente registros de
Anchoa y Sardina para construir los rangos de la interfaz.

Los sliders utilizan aproximadamente el 98 % central
de los valores observados, evitando que unos pocos
valores extremos generen escenarios poco representativos.

La localización se valida adicionalmente comprobando
que existan muestras CalCOFI muy próximas.

Las probabilidades mostradas corresponden a la
clasificación realizada por el modelo y no representan
directamente la cantidad real de peces presentes.
        """
    )


st.caption(
    "Datos CalCOFI · Modelo XGBoost · Clasificación de Anchoa y Sardina"
)
