# Importando as bibliotecas 
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
import numpy as np

sns.set(context='talk', style='ticks')

st.set_page_config(
     page_title="Previsão Renda",
     page_icon='https://cdn-icons-png.flaticon.com/512/950/950984.png',
     layout="wide",
)

st.write('# Análise exploratória da previsão de renda')

# Importando os dados:
renda = pd.read_csv("D:\\CURSOS\\2 - EBAC\\Ciêntista de Dados\\2 - Cientista de Dados\\17 - Métodos de analise\\Meu Projeto\\df_renda_clean.csv")
st.markdown('----')

# Introdução:
st.subheader("Entendendo a tendencia de renda para as demais vairaveis")
st.markdown("Esses dados são referentes as informações dos clientes como por exemplo; renda, sexo, posse_de_veiculo, posse_de_imovel, qtd_filhos e tipo_renda")
st.markdown('----')

# transformando os dados de csv para dataframe e mostrando na tela do aplicativo:
st.write('A seguir, você encontrará uma tabela contendo nossos dados:\nque serão utilizados ao longo de nossas análises:')
st.dataframe(renda)
st.markdown('----')


# Adicionando filtros:

# Extrair a lista de estados civis únicos
estados_civis_unicos = renda['estado_civil'].unique()

# Criar o filtro multiselect para os estados civis na barra lateral esquerda
estados_civis_selecionados = st.sidebar.multiselect('Selecione o Estado Civil', estados_civis_unicos)

# Aplicar o filtro
if estados_civis_selecionados:
    renda_filtrado = renda[renda['estado_civil'].isin(estados_civis_selecionados)]
    st.write(renda_filtrado)
else:
    st.write('Nenhum estado civil selecionado. Por favor, selecione um ou mais estados civis.')
st.markdown('----')


# Criando filtro para renda:

# Extrair a lista de tipos de renda únicos
tipos_renda_unicos = renda['tipo_renda'].unique()

# Criar o filtro multiselect para os tipos de renda na barra lateral esquerda
tipos_renda_selecionados = st.sidebar.multiselect('Selecione o Tipo de Renda', tipos_renda_unicos)

# Aplicar o filtro
if tipos_renda_selecionados:
    renda_filtrado = renda[renda['tipo_renda'].isin(tipos_renda_selecionados)]
    st.write(renda_filtrado)
else:
    st.write('Nenhum tipo de renda selecionado. Por favor, selecione um ou mais tipos de renda.')
st.markdown('----')

#Criando filtro para data

# Converter a coluna 'data_ref' para datetime
renda['data_ref'] = pd.to_datetime(renda['data_ref'])

# Criar o filtro de seletor de datas para 'data_ref' na barra lateral esquerda
data_ref_inicial, data_ref_final = st.sidebar.date_input('Selecione o intervalo de datas', [renda['data_ref'].min(), renda['data_ref'].max()])


#plots
fig, ax = plt.subplots(8,1,figsize=(10,70))
renda[['posse_de_imovel','renda']].plot(kind='hist', ax=ax[0])
st.write('## Gráficos ao longo do tempo')
sns.lineplot(x='data_ref',y='renda', hue='posse_de_imovel',data=renda, ax=ax[1])
ax[1].tick_params(axis='x', rotation=45)
sns.lineplot(x='data_ref',y='renda', hue='posse_de_veiculo',data=renda, ax=ax[2])
ax[2].tick_params(axis='x', rotation=45)
sns.lineplot(x='data_ref',y='renda', hue='qtd_filhos',data=renda, ax=ax[3])
ax[3].tick_params(axis='x', rotation=45)
sns.lineplot(x='data_ref',y='renda', hue='tipo_renda',data=renda, ax=ax[4])
ax[4].tick_params(axis='x', rotation=45)
sns.lineplot(x='data_ref',y='renda', hue='educacao',data=renda, ax=ax[5])
ax[5].tick_params(axis='x', rotation=45)
sns.lineplot(x='data_ref',y='renda', hue='estado_civil',data=renda, ax=ax[6])
ax[6].tick_params(axis='x', rotation=45)
sns.lineplot(x='data_ref',y='renda', hue='tipo_residencia',data=renda, ax=ax[7])
ax[7].tick_params(axis='x', rotation=45)
sns.despine()
st.pyplot(plt)

st.write('## Gráficos bivariada')
fig, ax = plt.subplots(7,1,figsize=(10,50))
sns.barplot(x='posse_de_imovel',y='renda',data=renda, ax=ax[0])
sns.barplot(x='posse_de_veiculo',y='renda',data=renda, ax=ax[1])
sns.barplot(x='qtd_filhos',y='renda',data=renda, ax=ax[2])
sns.barplot(x='tipo_renda',y='renda',data=renda, ax=ax[3])
sns.barplot(x='educacao',y='renda',data=renda, ax=ax[4])
sns.barplot(x='estado_civil',y='renda',data=renda, ax=ax[5])
sns.barplot(x='tipo_residencia',y='renda',data=renda, ax=ax[6])
sns.despine()
st.pyplot(plt)


#Machine Laerning
import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf

# Carregando os dados
df_renda = pd.read_csv(r"D:\CURSOS\2 - EBAC\Ciêntista de Dados\2 - Cientista de Dados\17 - Métodos de analise\Meu Projeto\previsao_de_renda.csv")

# Limpando os dados
df_clean_renda = df_renda.dropna()

# Calculando a matriz de correlação
df_dummies_1 = pd.get_dummies(df_clean_renda)
corr_df1 = df_dummies_1.corr()

# Primeiro modelo de Previsão

# Regressão multipla
reg = smf.ols('renda ~ sexo + posse_de_veiculo + posse_de_imovel + qtd_filhos + tipo_renda + educacao + estado_civil + tipo_residencia + idade + tempo_emprego + qt_pessoas_residencia', data=df_clean_renda).fit()

# Calculando os resíduos do modelo
df_clean_renda['res_log'] = reg.resid

# Definindo a variável resposta como os resíduos do modelo
df_clean_renda.loc[:, 'renda'] = reg.resid

# Lista das variáveis para os gráficos
variables = ['sexo', 'posse_de_veiculo', 'posse_de_imovel', 'qtd_filhos', 'tipo_renda', 'educacao', 'estado_civil', 'tipo_residencia', 'idade', 'tempo_emprego', 'qt_pessoas_residencia']

# Identificar as correlações mais fortes com a variável:
strong_corr = corr_df1['renda'].abs().sort_values(ascending=False)
st.write("Correlação mais fortes com renda.")
st.write(strong_corr)

# Plotando as variáveis com correlação igual ou maior que 0.07
threshold = 0.07
strong_corr = strong_corr[strong_corr > threshold]
st.write("\nCorrelações mais fortes acima do limiar de", threshold)
st.write(strong_corr)


# Plotando os gráficos de dispersão com os resíduos
for var in variables:
    fig, ax = plt.subplots()
    sns.scatterplot(x=var, y='renda', data=df_clean_renda, alpha=.75, ax=ax)
    ax.axhline(y=0, color='r', linestyle='--')
    ax.set_title(f'Residuals vs {var}')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90)  # Rotacionando os rótulos do eixo x
    st.pyplot(fig)

# Plotando a clustermap
plt.figure()
sns.clustermap(corr_df1, figsize=(10, 10), center=0, cmap='coolwarm')
plt.title('Clustermap - Correlation Matrix')
plt.xticks(rotation=90)  # Rotacionando os rótulos do eixo x
st.pyplot()






