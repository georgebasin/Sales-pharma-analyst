import pandas as pd
import numpy as np

import anthropic
client = anthropic.Anthropic(api_key="TU_API_KEY")
response = client.messages.create(
    model="claude-3-5-sonnet-20241022",
    max_tokens=2000,
    messages=[{"role": "user", "content": prompt_usuario}]
)
print(response.content[0].text)

def procesar_datos_ventas(file_path):
    """Paso 1: Parsea y limpia el CSV/Excel extrayendo las métricas clave."""
    df_raw = pd.read_csv(file_path, header=3)
    
    # Limpieza
    df = df_raw.dropna(subset=['ACCOUNT']).copy()
    df = df[df['ACCOUNT'] != 'ACCOUNT']
    
    meta_cols = ['ACCOUNT', 'CUSTOMER NAME', 'CITY', 'PROVINCE', 'REGION', 'KAM', 'PRODUCT']
    month_cols = [c for c in df.columns if c not in meta_cols and c != 'YTD']
    
    for col in month_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        
    df['Total_Ventas'] = df[month_cols].sum(axis=1)
    
    # Comparativa últimos 12 meses (R12M) vs 12 meses anteriores (P12M)
    recent_12m = month_cols[-12:]
    prev_12m = month_cols[-24:-12]
    
    df['R12M'] = df[recent_12m].sum(axis=1)
    df['P12M'] = df[prev_12m].sum(axis=1)
    df['Diff_12M'] = df['R12M'] - df['P12M']
    
    # Agrupaciones clave
    hosp_summary = df.groupby('ACCOUNT')[['P12M', 'R12M', 'Diff_12M', 'Total_Ventas']].sum()
    prod_summary = df.groupby('PRODUCT')[['P12M', 'R12M', 'Diff_12M', 'Total_Ventas']].sum()
    region_summary = df.groupby('REGION')[['P12M', 'R12M', 'Diff_12M', 'Total_Ventas']].sum() if 'REGION' in df.columns else None
    
    top_creciendo = hosp_summary[hosp_summary['Diff_12M'] > 0].sort_values('Diff_12M', ascending=False).head(8)
    top_caida = hosp_summary[hosp_summary['Diff_12M'] < 0].sort_values('Diff_12M', ascending=True).head(8)
    
    # Construir un resumen compacto en texto para enviar al LLM
    resumen_texto = f"""
    MÉTRICAS CLAVE DEL DATASET:
    - Total Cuentas / Hospitales: {df['ACCOUNT'].nunique()}
    - Total Unidades Vendidas Histórico: {df['Total_Ventas'].sum():.0f}

    RESUMEN POR PRODUCTO (12m Recientes vs 12m Anteriores):
    {prod_summary.to_string()}

    TOP HOSPITALES EN CAÍDA (Mayor pérdida de volumen reciente):
    {top_caida.to_string()}

    TOP HOSPITALES EN CRECIMIENTO / RECUPERACIÓN:
    {top_creciendo.to_string()}
    """
    
    return resumen_texto

def generar_informe_ia(file_path):
    """Paso 2 y 3: Envía los datos procesados al LLM y obtiene el informe."""
    
    # 1. Extraer los datos procesados
    resumen_datos = procesar_datos_ventas(file_path)
    
    # 2. Definir el prompt para el modelo de IA
    prompt_sistema = """
    Eres un Director Comercial y Consultor Estratégico Senior experto en el sector hospitalario y farmacéutico.
    Tu objetivo es analizar los datos de ventas proporcionados y redactar un informe ejecutivo claro, riguroso y orientado a la acción.
    """
    
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
    
    # 3. Llamada a la API de OpenAI
    response = client.chat.completions.create(
        model="gpt-4o", # O puedes usar "gpt-4o-mini" para pruebas más económicas
        messages=[
            {"role": "system", "content": prompt_sistema},
            {"role": "user", "content": prompt_usuario}
        ],
        temperature=0.3 # Temperatura baja para análisis precisos y consistentes
    )
    
    return response.choices[0].message.content

# --- EJECUCIÓN ---
# informe_final = generar_informe_ia('HCO_MES 2.csv')
# print(informe_final)
