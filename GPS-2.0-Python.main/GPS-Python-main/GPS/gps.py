import osmnx as ox  # Biblioteca para baixar e manipular dados do OpenStreetMap
import networkx as nx  # Biblioteca para trabalhar com grafos e rotas
import matplotlib.pyplot as plt  # Biblioteca para gráficos e mapas
import contextily as ctx  # Biblioteca para adicionar mapa de fundo
import numpy as np
import random
from sklearn.neural_network import MLPRegressor
from warnings import filterwarnings

# Ignorar avisos irrelevantes de versões
filterwarnings('ignore')
# ==============================================================================
# MÓDULO DE REDES NEURAIS (RNA) - Previsão de Tráfego
# ==============================================================================
class PrevisorTrafegoRNA:
    def __init__(self):
        self.modelo = MLPRegressor(hidden_layer_sizes=(8, 4), max_iter=2000, random_state=42)
        self._treinar_modelo_ficticio()

    def _treinar_modelo_ficticio(self):
        """
        Treina a RNA com dados históricos simulados.
        Entrada: [Hora do Dia (0-24), Fim de Semana (0 ou 1), Chuva (0 ou 1)]
        Saída: Fator de Atraso (1.0 = normal, 2.0 = dobro do tempo)
        """
        X_treino = [
            [3, 0, 0], [8, 0, 0], [8, 0, 1],  # Madrugada livre, Rush manhã, Rush+Chuva
            [12, 0, 0], [18, 0, 0], [18, 0, 1], # Almoço, Rush tarde, Rush+Chuva
            [15, 1, 0], [23, 0, 0]              # Sábado tarde, Noite calma
        ]
        # y = Fator de lentidão
        y_treino = [
            1.0, 1.5, 1.8,
            1.2, 1.8, 2.2,
            1.1, 1.0
        ]
        self.modelo.fit(X_treino, y_treino)

    def prever_fator(self, hora, fds=0, chuva=0):
        entrada = [[hora, fds, chuva]]
        fator = self.modelo.predict(entrada)[0]
        return max(1.0, fator)

# ==============================================================================
# MÓDULO DE ALGORITMO GENÉTICO (AG) - Logística (Caixeiro Viajante)
# ==============================================================================
class OtimizadorLogisticaAG:
    def __init__(self, grafo, nos_entrega, populacao_tam=20, geracoes=50):
        self.grafo = grafo
        self.nos_entrega = nos_entrega  # Lista de nós [Origem, Destino1, Destino2...]
        self.populacao_tam = populacao_tam
        self.geracoes = geracoes

    def _distancia_euclidiana_estimada(self, rota):
        """Calcula distância em linha reta para ser rápido no AG"""
        dist = 0
        for i in range(len(rota) - 1):
            u, v = rota[i], rota[i+1]
            x1, y1 = self.grafo.nodes[u]['x'], self.grafo.nodes[u]['y']
            x2, y2 = self.grafo.nodes[v]['x'], self.grafo.nodes[v]['y']
            dist += ((x1 - x2)**2 + (y1 - y2)**2)**0.5
        return dist

    def resolver(self):
        origem = self.nos_entrega[0]
        destinos = self.nos_entrega[1:]
        
        # Criar população inicial (várias ordens de entrega aleatórias)
        populacao = []
        for _ in range(self.populacao_tam):
            rota_aleatoria = destinos[:]
            random.shuffle(rota_aleatoria)
            populacao.append([origem] + rota_aleatoria) # Sempre começa na origem

        # Evolução
        for _ in range(self.geracoes):
            # Ordenar por aptidão (menor distância é melhor)
            populacao.sort(key=self._distancia_euclidiana_estimada)
            
            # Seleção (Top 50%)
            melhores = populacao[:self.populacao_tam // 2]
            nova_geracao = melhores[:]
            
            # Cruzamento e Mutação para preencher o resto
            while len(nova_geracao) < self.populacao_tam:
                pai1, pai2 = random.sample(melhores, 2)
                
                # Crossover (preservar origem no índice 0)
                corte = len(pai1) // 2
                genes_pai1 = pai1[1:corte]
                genes_pai2 = [x for x in pai2[1:] if x not in genes_pai1]
                filho = [origem] + genes_pai1 + genes_pai2
                
                # Mutação (troca de posição de duas entregas)
                if random.random() < 0.2 and len(filho) > 3:
                    idx1, idx2 = random.sample(range(1, len(filho)), 2)
                    filho[idx1], filho[idx2] = filho[idx2], filho[idx1]
                
                nova_geracao.append(filho)
            
            populacao = nova_geracao

        # Retorna a melhor rota encontrada
        return populacao[0]
# ==============================================================================
# MÓDULO LÓGICA FUZZY - Avaliação de Qualidade
# ==============================================================================
class AvaliadorFuzzy:
    def avaliar(self, distancia_km, tempo_min):
        """
        Regras Fuzzy Simplificadas:
        - Distância (Curta, Média, Longa)
        - Tempo (Rápido, Médio, Demorado)
        -> Saída: Score (0-10) e Classificação
        """
        # Fuzzificação
        d_curta = max(0, min(1, (5 - distancia_km) / 5))
        d_longa = max(0, min(1, (distancia_km - 5) / 10))
        
        t_rapido = max(0, min(1, (15 - tempo_min) / 15))
        t_demorado = max(0, min(1, (tempo_min - 10) / 20))

        # Regras de Inferência (Score base)
        score = 5.0 # Começa neutro
        
        if d_curta > 0.5 and t_rapido > 0.5:
            score += 4.0 # Excelente
        elif d_longa > 0.5 or t_demorado > 0.5:
            score -= 3.0 # Ruim
        else:
            score += 1.0 # Bom/Regular

        # Defuzzificação (Limites)
        score = max(0, min(10, score))
        
        if score >= 8: label = "Excelente 🌟"
        elif score >= 6: label = "Boa ✅"
        elif score >= 4: label = "Regular ⚠️"
        else: label = "Ruim/Cansativa 🛑"
        
        return score, label
# ==============================================================================
# PROGRAMA PRINCIPAL
# ==============================================================================
def main():
    print("--- 🚚 SISTEMA INTELIGENTE DE LOGÍSTICA URBANA ---")
    print("Técnicas: A* (Rota), RNA (Tráfego), Genético (Ordem), Fuzzy (Avaliação)\n")

    # --- CONFIGURAÇÕES INICIAIS (COM INPUT DO USUÁRIO) ---
    
    # 1. Solicitar Origem
    origem_str = input("📍 Digite o Endereço de Origem (ex: Av. Tuparendi, Santa Rosa, RS): ")
    if not origem_str: 
        origem_str = "Av. Tuparendi, Santa Rosa, RS, Brazil" # Default se der enter vazio
        print(f"   (Usando padrão: {origem_str})")

    # 2. Solicitar Destinos (Loop para vários pontos)
    destinos_str = []
    print("\n📦 Digite os endereços de entrega (pressione ENTER sem digitar para encerrar):")
    
    contador = 1
    while True:
        end = input(f"   > Destino {contador}: ")
        if not end: # Se for vazio, para
            break
        destinos_str.append(end)
        contador += 1

    # Validação simples caso o usuário não digite nada
    if not destinos_str:
        print("⚠️ Nenhum destino inserido! Usando destinos de teste padrão...")
        destinos_str = [
            "Av. Rio Branco, Santa Rosa, RS, Brazil",
            "Av. America, Santa Rosa, RS, Brazil",
            "Rua Santa Rosa, Santa Rosa, RS, Brazil"
        ]

    # Solicitar Hora para a RNA
    try:
        hora_input = input("\n🕒 Digite a hora da entrega (0-23) [Enter para 18h]: ")
        hora_atual = float(hora_input) if hora_input else 18.0
    except ValueError:
        hora_atual = 18.0

    print(f"\n📝 Resumo da Missão:")
    print(f"   Origem: {origem_str}")
    print(f"   Total de Entregas: {len(destinos_str)}")
    print(f"   Horário: {hora_atual}h")

    # --- O RESTANTE DO CÓDIGO PERMANECE IGUAL ---
    
    # 1. Baixar Grafo
    print("\n🌎 Baixando mapa da região (pode demorar um pouco)...")
    # ... (O resto do código segue exatamente como na versão anterior) ...
    # Só certifique-se de copiar o resto da lógica (coords_origem, grafo, RNA, AG, A*, Fuzzy) aqui para baixo.
    
    coords_origem = ox.geocode(origem_str)
    grafo = ox.graph_from_point(coords_origem, dist=4000, network_type="drive")
    
    grafo = ox.add_edge_speeds(grafo)
    grafo = ox.add_edge_travel_times(grafo)

    # 2. APLICAR BARREIRAS 
    barreiras = [
        (-27.851760, -54.491188), (-27.847103, -54.483097),
        (-27.863107, -54.470706), (-27.863487, -54.466207)
    ]
    barreiras_nodes = [ox.distance.nearest_nodes(grafo, lon, lat) for lat, lon in barreiras]
    for node in barreiras_nodes:
        if node in grafo:
            grafo.remove_edges_from(list(grafo.edges(node)))
    print("🚧 Barreiras aplicadas ao mapa.")

    # 3. RNA: PREVISÃO DE TRÁFEGO
    print("\n🧠 [RNA] Analisando condições de tráfego...")
    rna = PrevisorTrafegoRNA()
    fator_transito = rna.prever_fator(hora=hora_atual, chuva=1)
    print(f"   -> Fator de Atraso Previsto: {fator_transito:.2f}x")

    for u, v, data in grafo.edges(data=True):
        tempo_original = data.get('travel_time', 1)
        data['travel_time'] = tempo_original * fator_transito

    # 4. PREPARAR PONTOS PARA O AG
    print("\n🧬 [Genético] Calculando melhor ordem de entrega...")
    no_origem = ox.distance.nearest_nodes(grafo, coords_origem[1], coords_origem[0])
    nos_destinos = []
    
    # Validação de geocoding para evitar erros se o endereço não for encontrado
    for dest in destinos_str:
        try:
            c = ox.geocode(dest)
            nos_destinos.append(ox.distance.nearest_nodes(grafo, c[1], c[0]))
        except Exception as e:
            print(f"   ⚠️ Endereço não encontrado e ignorado: {dest}")

    if not nos_destinos:
        print("❌ Nenhum destino válido encontrado. Encerrando.")
        return

    todos_nos = [no_origem] + nos_destinos
    
    # Rodar Algoritmo Genético
    ag = OtimizadorLogisticaAG(grafo, todos_nos)
    melhor_ordem = ag.resolver()
    
    # Mapear IDs dos nós de volta para nomes (apenas para exibição se quiseres, opcional)
    print(f"   -> Rota Otimizada definida com sucesso!")

    # 5. A*: CALCULAR ROTA FINAL
    print("\n⭐ [A*] Calculando rota detalhada...")
    rota_completa = []
    distancia_total_km = 0
    tempo_total_min = 0

    for i in range(len(melhor_ordem) - 1):
        u, v = melhor_ordem[i], melhor_ordem[i+1]
        try:
            caminho = nx.astar_path(grafo, u, v, weight='travel_time')
            rota_completa.append(caminho)
            
            # Calcular distância e tempo iterando pelas arestas do caminho
            d_m = 0
            t_s = 0
            for j in range(len(caminho) - 1):
                u_edge, v_edge = caminho[j], caminho[j+1]
                edge_data = grafo.get_edge_data(u_edge, v_edge)
                if not edge_data:
                    continue
                # edge_data pode ser um dict de arestas paralelas (key -> attr)
                # pegar o primeiro atributo disponível
                if isinstance(edge_data, dict):
                    first_attr = next(iter(edge_data.values()))
                    d_m += first_attr.get('length', 0)
                    t_s += first_attr.get('travel_time', 0)
                else:
                    d_m += edge_data.get('length', 0)
                    t_s += edge_data.get('travel_time', 0)

            distancia_total_km += d_m / 1000
            tempo_total_min += t_s / 60
        except nx.NetworkXNoPath:
            print(f"   ❌ Não há caminho entre dois pontos da rota.")

    print(f"   -> Distância Total: {distancia_total_km:.2f} km")
    print(f"   -> Tempo Estimado: {tempo_total_min:.1f} min")

    # 6. FUZZY: AVALIAÇÃO FINAL
    print("\n📊 [Fuzzy] Avaliando qualidade da rota...")
    fuzzy = AvaliadorFuzzy()
    nota, qualidade = fuzzy.avaliar(distancia_total_km, tempo_total_min)
    print(f"   -> Classificação: {qualidade} (Nota: {nota:.1f}/10)")

    # 7. VISUALIZAÇÃO
    print("\n🖼️ Gerando mapa final...")
    if rota_completa:
        cores = ['blue', 'red', 'green', 'purple', 'orange', 'cyan'][:len(rota_completa)]
        # Se houver mais rotas que cores, repete a última
        while len(cores) < len(rota_completa): cores.append('black')

        fig, ax = ox.plot_graph_routes(
            grafo, rota_completa,
            route_colors=cores,
            route_linewidth=4, node_size=0, show=False, close=False
        )
        
        ctx.add_basemap(ax, crs=grafo.graph['crs'], source=ctx.providers.CartoDB.Positron)
        
        info_text = (
            f"Logística Inteligente\n"
            f"Tráfego (RNA): {fator_transito:.2f}x\n"
            f"Distância: {distancia_total_km:.2f} km\n"
            f"Tempo: {tempo_total_min:.0f} min\n"
            f"Avaliação: {qualidade}"
        )
        ax.text(0.02, 0.95, info_text, transform=ax.transAxes, 
                bbox=dict(facecolor='white', alpha=0.8), fontsize=10)

        # Adicionar apenas marcadores para Origem e Destinos (sem nomes no mapa)
        try:
            label_nodes = [no_origem] + nos_destinos
            for node in label_nodes:
                if node in grafo.nodes:
                    x = grafo.nodes[node].get('x')
                    y = grafo.nodes[node].get('y')
                    if x is None or y is None:
                        continue
                    ax.scatter([x], [y], c='black', s=30, zorder=6)
        except Exception:
            # Não bloquear a plotagem se algo falhar aqui
            pass

        # Criar legenda separada (fora do mapa) mostrando cor -> trecho
        try:
            from matplotlib.lines import Line2D
            # labels para cada trecho: Origem -> Destino1, Destino1 -> Destino2, ...
            trecho_labels = []
            for i in range(len(rota_completa)):
                if i == 0:
                    start_label = 'Origem'
                else:
                    start_label = f'Destino {i}'
                end_label = f'Destino {i+1}'
                trecho_labels.append(f"{start_label} → {end_label}")

            legend_handles = [Line2D([0], [0], color=cores[i], lw=4) for i in range(len(rota_completa))]

            # Posicionar legenda discreta no canto superior direito sem atrapalhar o mapa
            try:
                # pequena legenda dentro do mapa, semi-transparente
                legend = ax.legend(legend_handles, trecho_labels,
                                   loc='upper right', fontsize=8,
                                   frameon=True, framealpha=0.75,
                                   borderpad=0.4, handletextpad=0.6)
                legend.get_frame().set_linewidth(0.5)
            except Exception:
                # fallback: legenda externa à direita
                try:
                    fig.subplots_adjust(right=0.72)
                    ax.legend(legend_handles, trecho_labels, loc='center left', bbox_to_anchor=(1.02, 0.5), fontsize=9, frameon=True)
                except Exception:
                    pass
        except Exception:
            pass
        plt.savefig("Rota_Inteligente.png", dpi=300)
        print("✅ Mapa salvo como 'Rota_Inteligente.png'")
        plt.show()
    else:
        print("⚠️ Nenhuma rota foi gerada para plotar.")

if __name__ == "__main__":
    main()