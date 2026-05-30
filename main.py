# import preprocessing as pp;
# import MLP_network as mlp;
import modulos.traning_evaluation as te;
import modulos.gerar_grafico as gg;

def run():
    # te.teste_inicial()
    # te.teste_final()
    
    gg.gerar_grafico_colunas_f1()
    #gg.gerar_boxplots_folds()
    #gg.gerar_grafico_evolucao_campeao()

if __name__ == "__main__": 
    
    run()