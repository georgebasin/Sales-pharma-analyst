import os
import pandas as pd
import anthropic

# La API key se lee de la variable de entorno ANTHROPIC_API_KEY
client = anthropic.Anthropic()

MODEL = "claude-sonnet-5-5"

META_COLS = ['ACCOUNT', 'CUSTOMER NAME', 'CITY', 'PROVINCE', 'REGION', 'KAM', 'PRODUCT']


def procesar_datos_ventas(file_path, sep=",", encoding="utf-8-sig", header_row=3, sheet_name=0):
    """Paso 1: Parsea y limpia el CSV o Excel extrayendo las métricas clave."""
    ext = os.path.splitext(file_path)[1].lower()

    if ext in (".xlsx", ".xlsm", ".xls"):
        df_raw = pd.read_excel(file_path, header=header_row, sheet_name=sheet_name)
    elif ext in (".csv", ".txt"):
        df_raw = pd.read_csv(file_path, header=header_row, sep=sep, encoding=encoding)
    else:
        raise ValueError(f"Formato no soportado: {ext}. Usa .csv, .xlsx, .xlsm o .xls")

    # Limpia nombres de columnas (las fechas de Excel se muestran como "ene-24")
    df_raw.columns = [
        c.strftime("%b-%y") if hasattr(c, "strftime") else str(c).strip()
        for c in df_raw.columns
    ]

    if 'ACCOUNT' not in df_raw.columns:
        raise ValueError(
            f"No se encuentra la columna 'ACCOUNT'. Columnas detectadas: {list(df_raw.columns)}. "
            "Revisa header_row, sep, encoding o sheet_name."
        )

    # Limpieza
    df = df_raw.dropna(subset=['ACCOUNT']).copy()
    df = df[df['ACCOUNT'] != 'ACCOUNT']

    month_cols = [c for c in df.columns if c not in META_COLS and c != 'YTD']
    if len(month_cols) < 24:
        raise ValueError(
            f"Se necesitan al menos 24 columnas de meses para comparar R12M vs P12M; "
            f"solo hay {len(month_cols)}: {month_cols}"
        )

    for col in month_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    df['Total_Ventas'] = df[month_cols].sum(axis=1)

    # Últimos 12 meses (R12M) vs 12 meses anteriores (P12M)
    recent_12m = month_cols[-12:]
    prev_12m = month_cols[-24:-12]

    df['R12M'] = df[recent_12m].sum(axis=1)
    df['P12M'] = df[prev_12m].sum(axis=1)
    df['Diff_12M'] = df['R12M'] - df['P12M']

    # Agrupaciones clave
    cols_sum = ['P12M', 'R12M', 'Diff_12M', 'Total_Ventas']
    group_hosp = ['ACCOUNT', 'CUSTOMER NAME'] if 'CUSTOMER NAME' in df.columns else ['ACCOUNT']

    hosp_summary = df.groupby(group_hosp)[cols_sum].sum()
    prod_summary = df.groupby('PRODUCT')[cols_sum].sum() if 'PRODUCT' in df.columns else None
    region_summary = df.groupby('REGION')[cols_sum].sum() if 'REGION' in df.columns else None

    top_creciendo = hosp_summary[hosp_summary['Diff_12M'] > 0].sort_values('Diff_12M', ascending=False).head(8)
    top_caida = hosp_summary[hosp_summary['Diff_12M'] < 0].sort_values('Diff_12M', ascending=True).head(8)

    resumen_texto = f"""
MÉTRICAS CLAVE DEL DATASET:
- Total Cuentas / Hospitales: {df['ACCOUNT'].nunique()}
- Total Unidades Vendidas Histórico: {df['Total_Ventas'].sum():.0f}
- Periodo reciente (R12M): {recent_12m[0]} a {recent_12m[-1]}
- Periodo anterior (P12M): {prev_12m[0]} a {prev_12m[-1]}

RESUMEN POR PRODUCTO (12m recientes vs 12m anteriores):
{prod_summary.to_string() if prod_summary is not None else 'No disponible'}

RESUMEN POR REGIÓN:
{region_summary.to_string() if region_summary is not None else 'No disponible'}

TOP HOSPITALES EN CAÍDA (mayor pérdida de volumen reciente):
{top_caida.to_string()}

TOP HOSPITALES EN CRECIMIENTO / RECUPERACIÓN:
{top_creciendo.to_string()}
"""
    return resumen_texto


def generar_informe_ia(file_path, **kwargs):
    """Paso 2 y 3: Envía los datos procesados al LLM y obtiene el informe."""
    resumen_datos = procesar_datos_ventas(file_path, **kwargs)

    prompt_sistema = (
        "Eres un Director Comercial y Consultor Estratégico Senior experto en el sector "
        "hospitalario y farmacéutico. Tu objetivo es analizar los datos de ventas proporcionados "
        "y redactar un informe ejecutivo claro, riguroso y orientado a la acción."
    )

    prompt_usuario = f"""
Analiza la siguiente información agregada de ventas hospitalarias y genera un informe estructurado.

DATOS DEL EXCEL PROCESADO:
{resumen_datos}

ESTRUCTURA REQUERIDA PARA EL INFORME:
1. **Resumen Ejecutivo**: Breve síntesis del estado global del negocio.
2. **Análisis de Tendencias por Producto**: Qué líneas están traccionando y cuáles caen.
3. **Diagnóstico de Cuentas (Hospitales)**:
   - Principales focos de riesgo (cuentas en decrecimiento acelerado).
   - Cuentas con buen desempeño o recuperación.
4. **Plan de Acción y Recomendaciones Comerciales**: 3-4 acciones concretas que el equipo de ventas / KAMs debe ejecutar inmediatamente.

Redacta en un tono profesional, conciso y directo a la alta dirección.
"""

    response = client.messages.create(
        model=MODEL,
        max_tokens=2000,
        temperature=0.3,  # Baja, para análisis consistentes
        system=prompt_sistema,
        messages=[{"role": "user", "content": prompt_usuario}],
    )

    return response.content[0].text


if __name__ == "__main__":
    # Excel:
    informe_final = generar_informe_ia('HCO_MES 2.xlsx')
    # Excel con hoja concreta: generar_informe_ia('HCO_MES 2.xlsx', sheet_name="Ventas")
    # CSV con punto y coma:    generar_informe_ia('HCO_MES 2.csv', sep=";")
    print(informe_final)
