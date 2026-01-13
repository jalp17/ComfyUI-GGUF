import os
import sys
import requests
import shutil
import subprocess
from huggingface_hub import HfApi, create_repo
# ==========================================
# 1. CONFIGURACIÓN (Editado: QUANT_TYPE retirado de os.getenv)
# ==========================================
class Config:
    # Credenciales secretas (Siguen siendo de entorno por seguridad)
    HF_TOKEN = os.getenv("HF_TOKEN")
    HF_USER = os.getenv("HF_USER")
    CIVITAI_API_KEY = os.getenv("CIVITAI_API_KEY")
    
    # --- CONFIGURACIÓN DE CALIDAD (Local) ---
    # Define aquí las calidades que quieres generar. 
    # Ejemplo: ["Q8_0", "Q6_K", "Q5_K_S", "Q4_K_S"]
    QUANTS_TO_GENERATE = ["Q8_0", "Q6_K", "Q5_K_S", "Q4_K_S"]
    
    # Rutas en partición grande (/tmp)
    BASE_TEMP_DIR = "/tmp/factory"
    INPUT_DIR = os.path.join(BASE_TEMP_DIR, "input")
    OUTPUT_DIR = os.path.join(BASE_TEMP_DIR, "output")
    
    # Herramientas locales
    CONVERT_SCRIPT = "tools/convert.py"
    QUANTIZE_BIN = "./llama.cpp/llama-quantize"

# ==========================================
# 2. DOCUMENTACIÓN (README dinámico)
# ==========================================
class Documentation:
    @staticmethod
    def generate_readme(meta, quant_list):
        quants_str = "\n".join([f"- **{q}**: Versión comprimida en {q}." for q in quant_list])
        return f"""# {meta['name']} - Colección GGUF

Este repositorio contiene versiones optimizadas en formato **GGUF** del modelo original [{meta['name']}](https://civitai.com/models/{meta['id']}).

## 👤 Créditos y Autoría
- **Autor Original:** [{meta['author']}](https://civitai.com/user/{meta['author']})
- **Proceso de Cuantización:** [{Config.HF_USER}](https://huggingface.co/{Config.HF_USER})

> **Aviso de Autoría:** Este modelo no es de mi creación. Yo únicamente he realizado la cuantización (compresión). Todo el mérito pertenece al autor original.

## 📊 Versiones Disponibles
{quants_str}

## ⚙️ Detalles Técnicos
- **Arquitectura Base:** {meta['baseModel']}
- **Herramientas utilizadas:** [llama.cpp](https://github.com/ggerganov/llama.cpp) y [ComfyUI-GGUF](https://github.com/city96/ComfyUI-GGUF)

## 📜 Licencia y Fuente
Consulte la [página original del modelo en Civitai]({meta['id']}) para licencias y términos de uso.

## 📝 Descripción del Autor
{meta['description']}
"""

# ==========================================
# 3. CLIENTE CIVITAI
# ==========================================
class CivitaiClient:
    @staticmethod
    def get_metadata(url):
        parts = url.strip().split('/')
        model_id = next((parts[i+1] for i, p in enumerate(parts) if p == 'models'), None)
        if not model_id: return None
        
        headers = {"Authorization": f"Bearer {Config.CIVITAI_API_KEY}"}
        resp = requests.get(f"https://civitai.com/api/v1/models/{model_id}", headers=headers)
        if resp.status_code != 200: return None
        
        data = resp.json()
        if data['type'] != 'Checkpoint': return None
        
        ver = data['modelVersions'][0]
        return {
            "id": model_id,
            "name": data['name'],
            "author": data.get('creator', {}).get('username', 'Unknown'),
            "download_url": f"{ver['downloadUrl']}?token={Config.CIVITAI_API_KEY}",
            "description": data.get("description", ""),
            "baseModel": ver.get("baseModel", "SDXL")
        }

# ==========================================
# 4. ORQUESTADOR PRINCIPAL
# ==========================================
class Orchestrator:
    def __init__(self):
        self.api = HfApi()
        os.makedirs(Config.INPUT_DIR, exist_ok=True)
        os.makedirs(Config.OUTPUT_DIR, exist_ok=True)

    def process_all(self):
        if not os.path.exists("links.txt"):
            print("❌ No existe links.txt")
            return

        with open("links.txt", "r") as f:
            links = [l.strip() for l in f if l.strip()]

        for link in links:
            meta = CivitaiClient.get_metadata(link)
            if not meta: continue
            self.process_single(meta)

    def process_single(self, meta):
        # Limpiar nombre para repo
        safe_name = meta['name'].lower().replace(" ", "-").replace("_", "-").replace("(", "").replace(")", "")
        repo_id = f"{Config.HF_USER}/{safe_name}-GGUF"
        
        # Rutas en /tmp
        input_file = os.path.join(Config.INPUT_DIR, "model.safetensors")
        temp_f16 = os.path.join(Config.OUTPUT_DIR, "temp_f16.gguf")

        print(f"\n🚀 --- PROCESANDO: {meta['name']} ---")
        
        # 1. Descarga
        subprocess.run(["wget", "-O", input_file, meta['download_url']], check=True)
        
        # 2. Conversión base (FP16)
        print("⚙️ Paso 1: Generando base FP16...")
        subprocess.run(["python3", Config.CONVERT_SCRIPT, "--src", input_file, "--dst", temp_f16], check=True)
        os.remove(input_file) # Borrar safetensors (libera 6GB)

        # 3. Generar quants definidos en Config.QUANTS_TO_GENERATE
        for q in Config.QUANTS_TO_GENERATE:
            final_file = os.path.join(Config.OUTPUT_DIR, f"{safe_name}-{q}.gguf")
            print(f"⚖️ Paso 2: Cuantizando a {q}...")
            subprocess.run([Config.QUANTIZE_BIN, temp_f16, final_file, q], check=True)
            
            # Subir archivo individual
            print(f"⬆️ Subiendo {q} a Hugging Face...")
            create_repo(repo_id=repo_id, token=Config.HF_TOKEN, exist_ok=True)
            self.api.upload_file(path_or_fileobj=final_file, path_in_repo=os.path.basename(final_file), repo_id=repo_id)
            os.remove(final_file) # Borrar archivo tras subir (libera 4-8GB)

        # 4. Generar y subir README
        readme = Documentation.generate_readme(meta, Config.QUANTS_TO_GENERATE)
        self.api.upload_file(path_or_fileobj=readme.encode(), path_in_repo="README.md", repo_id=repo_id)

        # 5. Limpieza final de caché y temporales
        os.remove(temp_f16)
        shutil.rmtree(os.path.expanduser("~/.cache/huggingface/hub"), ignore_errors=True)
        print(f"✅ Repositorio completado: https://huggingface.co/{repo_id}")

if __name__ == "__main__":
    Orchestrator().process_all()