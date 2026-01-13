import os
import sys
import requests
import shutil
import subprocess
from huggingface_hub import HfApi, create_repo

# ==========================================
# 1. CONFIGURACIÓN Y CONSTANTES
# ==========================================
class Config:
    HF_TOKEN = os.getenv("HF_TOKEN")
    HF_USER = os.getenv("HF_USER")
    CIVITAI_API_KEY = os.getenv("CIVITAI_API_KEY")
    QUANT_TYPE = os.getenv("QUANT_TYPE", "Q5_K_S")
    
    # Rutas en partición grande (/tmp)
    BASE_TEMP_DIR = "/tmp/factory"
    INPUT_DIR = os.path.join(BASE_TEMP_DIR, "input")
    OUTPUT_DIR = os.path.join(BASE_TEMP_DIR, "output")
    
    # Rutas de herramientas locales
    CONVERT_SCRIPT = "tools/convert.py"
    QUANTIZE_BIN = "./llama.cpp/llama-quantize"

# ==========================================
# 2. CLIENTE API CIVITAI
# ==========================================
class CivitaiClient:
    @staticmethod
    def get_metadata(url):
        parts = url.strip().split('/')
        model_id = next((parts[i+1] for i, p in enumerate(parts) if p == 'models'), None)
        
        if not model_id: return None
        
        headers = {"Authorization": f"Bearer {Config.CIVITAI_API_KEY}"}
        resp = requests.get(f"https://civitai.com/api/v1/models/{model_id}", headers=headers)
        
        if resp.status_code != 200:
            print(f"❌ Error API Civitai: {resp.status_code}")
            return None
        
        data = resp.json()
        if data['type'] != 'Checkpoint':
            print(f"⚠️ {data['name']} no es Checkpoint. Saltando...")
            return None

        ver = data['modelVersions'][0]
        return {
            "id": model_id,
            "name": data['name'],
            "author": data.get('creator', {}).get('username', 'Unknown'),
            "download_url": f"{ver['downloadUrl']}?token={Config.CIVITAI_API_KEY}",
            "description": data.get("description", "No description provided."),
            "baseModel": ver.get("baseModel", "Unknown"),
            "allowNoCredit": ver.get("allowNoCredit", True),
            "allowCommercial": ver.get("allowCommercialUse", "None"),
            "NSFW": data.get("nsfw", False),
            "tags": data.get("tags", [])
        }

# ==========================================
# 3. GENERADOR DE DOCUMENTACIÓN (README)
# ==========================================
class Documentation:
    @staticmethod
    def generate_readme(meta):
        tags_str = ", ".join(meta['tags'])
        nsfw_status = "⚠️ Contenido Adulto (NSFW)" if meta['NSFW'] else "✅ Contenido Seguro (SFW)"
        
        return f"""# {meta['name']} - GGUF (Quantized)

Este repositorio contiene una versión optimizada en formato **GGUF** del modelo [{meta['name']}](https://civitai.com/models/{meta['id']}).

---

## 👤 Información del Autor Original
- **Autor:** [{meta['author']}](https://civitai.com/user/{meta['author']})
- **Fuente Original:** [Civitai Model Page](https://civitai.com/models/{meta['id']})

> **Aviso de Autoría:** Este modelo no es de mi creación. Yo únicamente he realizado el proceso de cuantización para permitir su uso en hardware con recursos limitados mediante la arquitectura GGUF. Por favor, apoya al autor original en Civitai.

---

## ⚙️ Características del Modelo
- **Arquitectura Base:** {meta['baseModel']}
- **Tipo de Modelo:** Checkpoint / Stable Diffusion
- **Clasificación:** {nsfw_status}
- **Etiquetas:** {tags_str}

## 📊 Detalles de la Compresión (GGUF)
- **Método de Cuantización:** {Config.QUANT_TYPE}
- **Formato:** GGUF (Versión 2)
- **Herramientas utilizadas:** [llama.cpp](https://github.com/ggerganov/llama.cpp) y [ComfyUI-GGUF](https://github.com/city96/ComfyUI-GGUF)

---

## 📜 Licencia y Permisos
Basado en la información de Civitai:
- **Uso comercial:** {meta['allowCommercial']}
- **Crédito obligatorio:** {"Sí" if not meta['allowNoCredit'] else "No"}

*Para una comprensión total de las licencias, visite el link de la fuente original.*

---

## 📝 Descripción del Autor
{meta['description']}

---
**Convertido automáticamente por la Factoría de GGUF en GitHub Codespaces.**
"""

# ==========================================
# 4. MOTOR DE CONVERSIÓN
# ==========================================
class GGUFConverter:
    @staticmethod
    def run_conversion(input_path, output_f16, final_gguf):
        # 1. Convertir a FP16
        print("⚙️ Paso 1: Convirtiendo a FP16...")
        subprocess.run(["python3", Config.CONVERT_SCRIPT, "--src", input_path, "--dst", output_f16], check=True)
        os.remove(input_path) # Liberar 6GB
        
        # 2. Cuantizar
        print(f"⚖️ Paso 2: Cuantizando a {Config.QUANT_TYPE}...")
        subprocess.run([Config.QUANTIZE_BIN, output_f16, final_gguf, Config.QUANT_TYPE], check=True)
        os.remove(output_f16) # Liberar 6GB

# ==========================================
# 5. ORQUESTADOR PRINCIPAL
# ==========================================
class Orchestrator:
    def __init__(self):
        self.api = HfApi()
        os.makedirs(Config.INPUT_DIR, exist_ok=True)
        os.makedirs(Config.OUTPUT_DIR, exist_ok=True)

    def process_all(self):
        if not os.path.exists("links.txt"):
            print("❌ No se encontró links.txt")
            return

        with open("links.txt", "r") as f:
            links = [l.strip() for l in f if l.strip()]

        for link in links:
            try:
                self.process_single(link)
            except Exception as e:
                print(f"💥 Error fatal procesando {link}: {e}")
                self.cleanup_error()

    def process_single(self, link):
        meta = CivitaiClient.get_metadata(link)
        if not meta: return

        safe_name = meta['name'].lower().replace(" ", "-").replace("_", "-").replace("(", "").replace(")", "")
        repo_id = f"{Config.HF_USER}/{safe_name}-GGUF"
        
        # Rutas
        input_file = os.path.join(Config.INPUT_DIR, "model.safetensors")
        temp_f16 = os.path.join(Config.OUTPUT_DIR, "temp.gguf")
        final_file = os.path.join(Config.OUTPUT_DIR, f"{safe_name}-{Config.QUANT_TYPE}.gguf")

        print(f"\n🌟 --- INICIANDO: {meta['name']} ---")
        
        # Descarga
        print(f"📥 Descargando de Civitai...")
        subprocess.run(["wget", "-O", input_file, meta['download_url']], check=True)
        
        # Conversión
        GGUFConverter.run_conversion(input_file, temp_f16, final_file)
        
        # Hugging Face
        print(f"⬆️ Subiendo a Hugging Face: {repo_id}")
        create_repo(repo_id=repo_id, token=Config.HF_TOKEN, exist_ok=True)
        self.api.upload_file(
            path_or_fileobj=final_file,
            path_in_repo=os.path.basename(final_file),
            repo_id=repo_id,
            token=Config.HF_TOKEN
        )
        
        # README
        readme = Documentation.generate_readme(meta)
        self.api.upload_file(
            path_or_fileobj=readme.encode(),
            path_in_repo="README.md",
            repo_id=repo_id,
            token=Config.HF_TOKEN
        )

        # Limpieza
        if os.path.exists(final_file): os.remove(final_file)
        shutil.rmtree(os.path.expanduser("~/.cache/huggingface/hub"), ignore_errors=True)
        print(f"✨ ¡Éxito total! Modelo subido y disco limpio.")

    def cleanup_error(self):
        """Limpia todo en caso de que el script falle a mitad"""
        for d in [Config.INPUT_DIR, Config.OUTPUT_DIR]:
            for f in os.listdir(d):
                os.remove(os.path.join(d, f))

if __name__ == "__main__":
    factory = Orchestrator()
    factory.process_all()