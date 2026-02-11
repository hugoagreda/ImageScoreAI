import pandas as pd
from pathlib import Path
import cv2

# =====================
# CONFIG
# =====================

INPUT_CSV = Path("../data/datasets/interior_semantic.csv")
OUTPUT_CSV = Path("../data/datasets/interior_final_candidates.csv")

VISUAL_THRESHOLD = 0.70

HUMAN_LABELING = True
MAX_LABELS = 25            # 👈 cuántas imágenes quieres etiquetar
RANDOMIZE_LABELING = True  # 👈 orden aleatorio
LABEL_ONLY_NEW = False     # 👈 False = puedes re-etiquetar aunque ya tengan label

# =====================
# LOAD DATA INTELIGENTE
# =====================


def ask_label_limit(total_available, default_limit):
    print(
        f"\nCuantas imagenes quieres etiquetar? "
        f"(1-{total_available}, Enter={default_limit}, 0=todas)"
    )

    while True:
        user_input = input("Cantidad: ").strip()

        if user_input == "":
            return min(default_limit, total_available)

        if not user_input.isdigit():
            print("Entrada invalida. Escribe un numero entero.")
            continue

        value = int(user_input)

        if value == 0:
            return total_available

        if 1 <= value <= total_available:
            return value

        print(
            f"Numero fuera de rango. Usa un valor entre 1 y {total_available}, "
            "o 0 para todas."
        )

if OUTPUT_CSV.exists():

    print("🔁 Cargando dataset final existente...")
    df_final = pd.read_csv(OUTPUT_CSV)

else:

    print("🆕 Creando dataset final desde interior_semantic.csv...")
    df = pd.read_csv(INPUT_CSV)

    print(f"Total imágenes iniciales: {len(df)}")

    df_final = df[
        (df["indoor_score"] < VISUAL_THRESHOLD) &
        (df["has_semantic_interior"] == 1)
    ].copy()

    print(f"Interiores confirmados: {len(df_final)}")

    # =====================
    # QUALITY BUCKET AUTO
    # =====================

    def assign_quality(row):

        detected = str(row["detected_objects"])

        if detected.strip() == "":
            obj_count = 0
        else:
            obj_count = len(detected.split(","))

        if row["indoor_score"] < 0.45 and obj_count >= 2:
            return "good"

        if row["indoor_score"] < 0.60:
            return "medium"

        return "bad"

    df_final["quality_bucket"] = df_final.apply(assign_quality, axis=1)

    df_final["quality_bucket_human"] = ""

    df_final.to_csv(OUTPUT_CSV, index=False)

    print("\n✅ DATASET FINAL CREADO")
    print(f"Guardado en: {OUTPUT_CSV}")

    print("\n📊 Distribución quality_bucket:")
    print(df_final["quality_bucket"].value_counts())

# =====================
# HUMAN LABELING MODE
# =====================

if HUMAN_LABELING:

    print("\n🧠 Iniciando modo etiquetado humano...")

    if LABEL_ONLY_NEW:
        df_to_label = df_final[
            df_final["quality_bucket_human"].fillna("") == ""
        ]
    else:
        df_to_label = df_final.copy()

    print(f"Total disponibles para etiquetar: {len(df_to_label)}")

    if RANDOMIZE_LABELING:
        df_to_label = df_to_label.sample(frac=1).reset_index()

    if len(df_to_label) == 0:
        print("No hay imagenes disponibles para etiquetar.")
        max_labels = 0
    else:
        max_labels = ask_label_limit(len(df_to_label), MAX_LABELS)
        print(f"Se etiquetaran hasta {max_labels} imagenes.")

    labeled_count = 0

    for _, row_data in df_to_label.iterrows():

        if labeled_count >= max_labels:
            break

        idx = row_data["index"]
        row = df_final.loc[idx]

        img_path = row["image_path"]

        img = cv2.imread(img_path)

        if img is None:
            print(f"No se pudo abrir: {img_path}")
            continue

        cv2.imshow("Label Image", img)

        print("\nImagen:", img_path)
        print("Pulsa: 1=bad | 2=medium | 3=good | ESC=salir")

        key = cv2.waitKey(0)

        if key == 27:
            break
        elif key == ord("1"):
            df_final.at[idx, "quality_bucket_human"] = "bad"
        elif key == ord("2"):
            df_final.at[idx, "quality_bucket_human"] = "medium"
        elif key == ord("3"):
            df_final.at[idx, "quality_bucket_human"] = "good"
        else:
            print("Tecla no válida")
            continue

        labeled_count += 1

        df_final.to_csv(OUTPUT_CSV, index=False)

    cv2.destroyAllWindows()

    print(f"\n✅ Etiquetadas {labeled_count} imágenes")

# =====================
# FINAL LABEL FUSION
# =====================

print("\n🔄 Fusionando etiquetas AUTO + HUMAN...")

def merge_labels(row):

    human = str(row.get("quality_bucket_human", "")).strip()

    if human != "" and human != "nan":
        return human

    return row["quality_bucket"]

df_final["final_quality"] = df_final.apply(merge_labels, axis=1)

df_final.to_csv(OUTPUT_CSV, index=False)

print("\n✅ final_quality creada")

print("\n📊 Distribución FINAL:")
print(df_final["final_quality"].value_counts())

# =====================
# STATS EXTRA
# =====================

diff = df_final[
    (df_final["quality_bucket_human"] != "") &
    (df_final["quality_bucket_human"] != df_final["quality_bucket"])
]

print(f"\n🧠 Correcciones humanas detectadas: {len(diff)}")
