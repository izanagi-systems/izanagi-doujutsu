# SIAC - Sistema Inteligente de Análise de Caixas

O SIAC é um sistema de visão computacional projetado para monitorar caixas em um ambiente industrial. Ele utiliza dois modelos de inteligência artificial (YOLOv8) para garantir que nenhuma caixa saia da linha de produção com uma contagem de itens incompleta.

## Principais Funcionalidades

- **Detecção Robusta da Caixa (ROI):** Um modelo YOLO dedicado localiza a caixa principal, independentemente de variações de iluminação ou fundo.
- **Contagem Precisa de Itens:** Um segundo modelo YOLO conta os itens dentro da caixa detectada.
- **Lógica de Alarme Inteligente:** O sistema implementa uma máquina de estados que monitora o ciclo de vida da caixa (presente, ausente) e utiliza um histórico de contagens para evitar falsos alarmes causados por movimentos rápidos.
- **Flexibilidade:** O sistema pode ser executado com uma câmera ao vivo, um arquivo de vídeo ou uma imagem estática.

---

## Estrutura do Projeto

```
.
├── dataset/
│   ├── 1_item_counter/     # Dataset para o modelo de contagem de itens
│   │   ├── images/
│   │   ├── labels/
│   │   └── data.yaml
│   └── 2_roi_detector/       # Dataset para o modelo de detecção da caixa (ROI)
│       ├── images/
│       ├── labels/
│       └── data.yaml
├── runs/                     # Pasta gerada pelo YOLO com os modelos treinados
│   └── detect/
├── capturador_imagens.py     # Script para capturar novas imagens para o dataset
├── extrator_frames.py        # Script para extrair frames de um vídeo
├── requirements.txt          # Lista de dependências do projeto
├── train.py                  # Script flexível para treinar os modelos
├── verificador_caixas.py     # Script principal da aplicação
└── README.md                 # Este documento
```

---

## Guia de Instalação e Uso

Siga os passos abaixo para configurar e executar o projeto em uma nova máquina.

### 1. Instalação de Dependências

Certifique-se de ter o Python 3.8+ instalado. Em seguida, instale as dependências do projeto usando o `pip`:

```bash
pip install -r requirements.txt
```

### 2. Treinamento dos Modelos (Opcional)

O projeto já pode conter modelos pré-treinados na pasta `runs/`. Caso deseje treinar seus próprios modelos com novas imagens, siga os passos abaixo.

**a) Prepare os Datasets:**
- Coloque as imagens para contagem de itens em `dataset/1_item_counter/images`.
- Coloque as imagens para detecção da caixa (ROI) em `dataset/2_roi_detector/images`.
- Anote as imagens usando uma ferramenta como o [makesense.ai](https://www.makesense.ai/) e salve os arquivos `.txt` nas pastas `labels/` correspondentes.

**b) Execute o Script de Treinamento:**

O script `train.py` é flexível e permite treinar qualquer um dos modelos.

- **Para treinar o detector de ROI:**
  ```bash
  python train.py --dataset 2_roi_detector --epochs 150 --name meu_detector_roi
  ```

- **Para treinar o contador de itens:**
  ```bash
  python train.py --dataset 1_item_counter --epochs 200 --name meu_contador_itens
  ```

Após o treinamento, os melhores modelos (`best.pt`) estarão salvos em pastas dentro de `runs/detect/` (ex: `runs/detect/meu_detector_roi/weights/best.pt`).

### 3. Execução do Sistema Principal

Para rodar o sistema, você precisa primeiro configurar os caminhos para os modelos que ele deve usar.

**a) Configure os Caminhos no Código:**

Abra o arquivo `verificador_caixas.py` e localize a seção de configuração no final do arquivo (`if __name__ == "__main__":`).

**Altere as variáveis `MODELO_ITENS` e `MODELO_ROI` para apontar para os seus arquivos `best.pt` treinados.**

```python
# Exemplo de configuração em verificador_caixas.py

MODELO_ITENS = 'runs/detect/meu_contador_itens/weights/best.pt'
MODELO_ROI = 'runs/detect/meu_detector_roi/weights/best.pt'

# Configure o perfil da caixa
perfil = {
    'nome': 'Caixa Padrão',
    'itens_esperados': 12
}
```

**b) Execute o Verificador:**

Use os argumentos de linha de comando para escolher a fonte de vídeo.

- **Para usar a câmera ao vivo:**
  ```bash
  python verificador_caixas.py --source cam
  ```

- **Para usar um arquivo de vídeo:**
  ```bash
  python verificador_caixas.py --source "caminho/para/seu/video.mp4"
  ```

- **Para analisar uma única imagem:**
  ```bash
  python verificador_caixas.py --source "caminho/para/sua/imagem.jpg" --mode image
  ```

O sistema iniciará, detectando a caixa, contando os itens e disparando o alarme conforme a lógica implementada.
