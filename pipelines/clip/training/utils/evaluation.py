import argparse
import os
import random

import cv2
import matplotlib.pyplot as plt
import numpy as np
import sklearn
import torch
import umap
from PIL import Image
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor


def generate_embeddings(model_path: str, images_dir: str, embeddings_file_path: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model: CLIPModel = CLIPModel.from_pretrained(model_path).to(device).eval()
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14-336")

    image_paths = []
    embeddings = []

    valid_extensions = (".jpg", ".jpeg", ".png")
    files = [f for f in os.listdir(images_dir) if f.lower().endswith(valid_extensions)]

    for filename in tqdm(files, desc=f"Processing {images_dir}"):
        image_path = os.path.join(images_dir, filename)
        image = Image.open(image_path).convert("RGB")

        inputs = processor(images=image, return_tensors="pt").to(device)
        with torch.no_grad():
            image_embeds = model.get_image_features(**inputs)

        embedding = image_embeds[0].cpu().tolist()

        embeddings.append(embedding)
        image_paths.append(image_path)

    np.savez(
        embeddings_file_path, embeddings=np.array(embeddings), image_paths=image_paths
    )


def load_stored_embeddings(file_path):
    data = np.load(file_path, allow_pickle=True)
    return data["embeddings"], data["image_paths"]


def reduce_dimensionality_umap(embeddings: np.ndarray, n_components: int) -> np.ndarray:
    reducer = umap.UMAP(n_components=n_components, random_state=42)
    reduced_embeddings = reducer.fit_transform(X=embeddings)
    return reduced_embeddings


def apply_dbscan_clustering(
    embeddings: np.ndarray, dbscan_eps: float, dbscan_min_samples: int
) -> np.ndarray:
    dbscan = sklearn.cluster.DBSCAN(eps=dbscan_eps, min_samples=dbscan_min_samples)
    labels = dbscan.fit_predict(embeddings)
    return labels


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument(
        "--embeddings_file", type=str, required=True, help="File containing embeddings"
    )
    parser.add_argument(
        "--dbscan_eps",
        type=float,
        required=True,
        help="DBSCAN epsilon parameter",
        default=0.5,
    )
    parser.add_argument(
        "--dbscan_min_samples",
        type=int,
        required=True,
        help="DBSCAN min_samples parameter",
        default=5,
    )
    args = parser.parse_args()

    embeddings = load_stored_embeddings(args.embeddings_file)
    labels = apply_dbscan_clustering(
        embeddings, args.dbscan_eps, args.dbscan_min_samples
    )


def save_clustering_plots(
    reduced_embeddings: np.ndarray, cluster_labels: np.ndarray, results_dir: str
):
    """Sauvegarde le plot des clusters DBSCAN avec annotations pour chaque cluster."""
    os.makedirs(results_dir, exist_ok=True)

    unique_clusters = np.unique(cluster_labels)
    plt.figure(figsize=(10, 6))

    # 📌 Boucle pour afficher chaque cluster
    for cluster in unique_clusters:
        indices = np.where(cluster_labels == cluster)[0]
        if cluster == -1:
            # Affichage des outliers en gris, plus petits
            plt.scatter(
                reduced_embeddings[indices, 0],
                reduced_embeddings[indices, 1],
                c="gray",
                s=10,
                alpha=0.3,
                label="Outliers" if cluster == -1 else None,
            )
        else:
            # Affichage normal des clusters
            plt.scatter(
                reduced_embeddings[indices, 0],
                reduced_embeddings[indices, 1],
                label=f"Cluster {cluster}",
                s=30,
                alpha=0.7,
            )

            # Ajouter le numéro du cluster au centre du cluster
            cluster_center = np.mean(reduced_embeddings[indices], axis=0)
            plt.text(
                cluster_center[0],
                cluster_center[1],
                str(cluster),
                fontsize=12,
                weight="bold",
                bbox={"facecolor": "white", "alpha": 0.6, "edgecolor": "black"},
            )

    plt.xlabel("UMAP 1")
    plt.ylabel("UMAP 2")
    plt.title("Clusters détectés avec DBSCAN")
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.savefig(
        os.path.join(results_dir, "dbscan_clusters_annotated.png"), bbox_inches="tight"
    )
    plt.close()

    print(
        f"📊 Plot amélioré des clusters DBSCAN sauvegardé dans {results_dir}/dbscan_clusters_annotated.png"
    )


def save_cluster_images_plot(
    image_paths,
    cluster_labels,
    results_dir,
    max_images_per_cluster=25,
    grid_size=(5, 5),
):
    """
    Génère une visualisation par cluster où toutes les images du cluster sont affichées ensemble sous forme de grille.

    :param image_paths: Liste des chemins des images.
    :param cluster_labels: Liste des labels de clusters DBSCAN.
    :param results_dir: Dossier où sauvegarder les plots.
    :param max_images_per_cluster: Nombre max d'images à afficher par cluster.
    :param grid_size: Taille de la grille d'affichage (ex: (5, 5) pour 25 images max).
    """
    os.makedirs(results_dir, exist_ok=True)

    unique_clusters = np.unique(cluster_labels)

    for cluster in unique_clusters:
        if cluster == -1:
            continue  # Ignore les outliers pour cette fonction

        indices = np.where(cluster_labels == cluster)[0]
        if len(indices) == 0:
            continue

        selected_indices = random.sample(
            list(indices), min(max_images_per_cluster, len(indices))
        )

        fig, axes = plt.subplots(grid_size[0], grid_size[1], figsize=(10, 10))
        fig.suptitle(f"Cluster {cluster}", fontsize=14)

        for ax, idx in zip(axes.flatten(), selected_indices, strict=False):
            img = cv2.imread(image_paths[idx])
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            ax.imshow(img)
            ax.axis("off")

        # Supprimer les axes inutilisés
        for i in range(len(selected_indices), grid_size[0] * grid_size[1]):
            axes.flatten()[i].axis("off")

        plot_path = os.path.join(results_dir, f"cluster_{cluster}_plot.png")
        plt.savefig(plot_path, bbox_inches="tight")
        plt.close()

        print(f"📸 Plot du cluster {cluster} sauvegardé dans {plot_path}")


def save_outliers_images(
    image_paths, cluster_labels, results_dir, max_images=25, grid_size=(5, 5)
):
    """Affiche et sauvegarde une grille d'images des outliers détectés par DBSCAN."""
    os.makedirs(results_dir, exist_ok=True)

    outliers_indices = np.where(cluster_labels == -1)[0]
    if len(outliers_indices) == 0:
        print("🚨 Aucun outlier détecté. Aucun plot généré.")
        return

    # Sélectionner aléatoirement jusqu'à `max_images` outliers
    num_images = min(len(outliers_indices), max_images)
    selected_indices = random.sample(list(outliers_indices), num_images)

    fig, axes = plt.subplots(grid_size[0], grid_size[1], figsize=(10, 10))
    fig.suptitle("Outliers détectés par DBSCAN", fontsize=14)

    valid_images = 0
    for ax, idx in zip(axes.flatten(), selected_indices, strict=False):
        image_path = str(image_paths[idx])  # 🔹 Convertir en string
        img = cv2.imread(image_path)

        if img is None:
            print(f"❌ Erreur : Impossible de charger {image_path}")
            ax.axis("off")
            continue

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        ax.imshow(img)
        ax.axis("off")
        valid_images += 1

    # Désactiver les axes vides si moins d'images que la grille
    for i in range(valid_images, grid_size[0] * grid_size[1]):
        axes.flatten()[i].axis("off")

    outliers_path = os.path.join(results_dir, "outliers_images.png")
    plt.savefig(outliers_path, bbox_inches="tight")
    plt.close()

    print(f"📸 Visualisation des outliers sauvegardée dans {outliers_path}")
