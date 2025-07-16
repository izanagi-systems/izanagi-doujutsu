import os

# O diretório que contém os arquivos de rótulo .txt
labels_dir = 'dataset/1_item_counter/labels'

# Verifica se o diretório existe
if not os.path.isdir(labels_dir):
    print(f"Erro: O diretório '{labels_dir}' não foi encontrado.")
    exit()

print(f"Iniciando a correção dos rótulos no diretório: {labels_dir}")
arquivos_processados = 0

# Itera sobre todos os arquivos no diretório
for filename in os.listdir(labels_dir):
    if filename.endswith('.txt'):
        filepath = os.path.join(labels_dir, filename)
        
        linhas_modificadas = []
        modificado = False
        
        with open(filepath, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if not parts:
                    linhas_modificadas.append(line)
                    continue
                
                class_id = int(parts[0])
                
                # A lógica da troca:
                # Se a classe for 0 (era divisor), vira 1.
                # Se a classe for 1 (era item), vira 0.
                if class_id == 0:
                    new_class_id = 1
                    modificado = True
                elif class_id == 1:
                    new_class_id = 0
                    modificado = True
                else:
                    new_class_id = class_id # Mantém outras classes, se houver
                
                # Remonta a linha com o novo ID
                nova_linha = f"{new_class_id} {' '.join(parts[1:])}\n"
                linhas_modificadas.append(nova_linha)

        # Se o arquivo foi modificado, salva as alterações
        if modificado:
            with open(filepath, 'w') as f:
                f.writelines(linhas_modificadas)
        
        arquivos_processados += 1

print(f"Processo concluído. {arquivos_processados} arquivos verificados e corrigidos.")
