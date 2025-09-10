# IEStoPBRT

**IEStoPBRT** is a Python toolkit to convert any [IES LM-63 photometric file](https://store.ies.org/product/approved-method-ies-standard-file-format-for-the-electronic-transfer-of-photometric-data-and-related-information/?v=ab6c04006660) into a **goniometric image** (lat–long HDR/EXR) suitable for use with [PBRT-v4](https://pbrt.org/) (Physically Based Rendering Toolkit).  

The project was developed by **Angélica Vargas Chavarro** (Universidad Nacional de Colombia) under the supervision of **Professor Carlos Ureña Almagro** (Universidad de Granada).

---

## ✨ Features

- 📂 **IES parsing** (LM-63-95/02 standard compliant).  
- 🔄 **Symmetry expansion** (0–90° / 0–180° → full 0–360°).  
- 🎯 **Flexible resampling** to configurable lat–long maps (`width × height`).  
- 💡 **EXR output** with float32 precision for photometric fidelity.  
- 📊 **Metadata reporting**: lumens, candela multiplier, units, vertical/horizontal angle counts.  
- 🛠️ **PBRT-ready** output for `LightSource "goniometric"`.  

---

## 📦 Installation

Clone the repository:

```bash
git clone https://github.com/<your-username>/IEStoPBRT.git
cd IEStoPBRT
