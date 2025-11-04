import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# =====================================
# ALL HELPER FUNCTIONS
# =====================================

def get_sift_matches(img1_gray, img2_gray, ratio_thresh=0.75, max_matches=50):
    """Detect and match SIFT keypoints using Lowe’s ratio test, returning top matches."""
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(img1_gray, None)
    
    # --- BUG FIX ---
    # The original code had img1_gray here, which would match the image against itself.
    # It has been corrected to img2_gray.
    kp2, des2 = sift.detectAndCompute(img2_gray, None)
    # ---------------

    # Brute-force matcher with KNN
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    # Handle potential descriptor mismatch (e.g., no keypoints)
    if des1 is None or des2 is None or len(des1) < 2 or len(des2) < 2:
        return [], kp1, kp2
        
    knn_matches = bf.knnMatch(des1, des2, k=2)

    # Apply ratio test
    good_matches = []
    for m, n in knn_matches:
        if m.distance < ratio_thresh * n.distance:
            good_matches.append(m)

    # Sort and take top 50 matches
    good_matches = sorted(good_matches, key=lambda x: x.distance)
    good_matches = good_matches[:max_matches]

    return good_matches, kp1, kp2


def compute_homography(points_src, points_dst):
    """Compute 3x3 homography matrix using the DLT algorithm."""
    assert points_src.shape == points_dst.shape
    n = points_src.shape[0]
    if n < 4:
        raise ValueError("At least 4 point correspondences are required.")

    A = []
    for i in range(n):
        x, y = points_src[i]
        x_p, y_p = points_dst[i]
        A.append([-x, -y, -1, 0, 0, 0, x_p * x, x_p * y, x_p])
        A.append([0, 0, 0, -x, -y, -1, y_p * x, y_p * y, y_p])

    A = np.array(A)
    _, _, Vt = np.linalg.svd(A)
    H = Vt[-1, :].reshape(3, 3)
    H /= H[2, 2]
    return H


def create_mosaic(img1, img2, H):
    """Create mosaic by warping img1 into img2’s coordinate frame."""
    h2, w2 = img2.shape[:2]
    h1, w1 = img1.shape[:2]

    # Corners of img1
    corners_img1 = np.float32([[0, 0], [0, h1], [w1, h1], [w1, 0]]).reshape(-1, 1, 2)
    warped_corners = cv2.perspectiveTransform(corners_img1, H)

    # Combine corners of both images
    corners_img2 = np.float32([[0, 0], [0, h2], [w2, h2], [w2, 0]]).reshape(-1, 1, 2)
    all_corners = np.concatenate((warped_corners, corners_img2), axis=0)

    [xmin, ymin] = np.int32(all_corners.min(axis=0).ravel() - 0.5)
    [xmax, ymax] = np.int32(all_corners.max(axis=0).ravel() + 0.5)

    translation = [-xmin, -ymin]
    T = np.array([[1, 0, translation[0]], [0, 1, translation[1]], [0, 0, 1]])

    # Warp img1 into mosaic
    mosaic_w, mosaic_h = xmax - xmin, ymax - ymin
    img1_warped = cv2.warpPerspective(img1, T @ H, (mosaic_w, mosaic_h))

    # Overlay img2
    mosaic = img1_warped.copy()
    
    # Create a mask for img2 to avoid overwriting black areas with black
    img2_mask = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    img2_mask = cv2.threshold(img2_mask, 1, 255, cv2.THRESH_BINARY)[1]

    # Place img2 onto the mosaic
    mosaic[translation[1]:translation[1]+h2, translation[0]:translation[0]+w2] = \
        cv2.bitwise_or(
            mosaic[translation[1]:translation[1]+h2, translation[0]:translation[0]+w2],
            img2,
            mask=img2_mask
        )

    return mosaic

def stitch_three_images(imgA, imgB, imgC, nameA, nameB, nameC, output_dir):
    """
    Performs a 2-stage stitch: (A + B) + C
    Returns the final mosaic, or None if it fails.
    """
    print(f"\n--- Stitching Order: ({nameA} + {nameB}) + {nameC} ---")
    grayA = cv2.cvtColor(imgA, cv2.COLOR_BGR2GRAY)
    grayB = cv2.cvtColor(imgB, cv2.COLOR_BGR2GRAY)
    grayC = cv2.cvtColor(imgC, cv2.COLOR_BGR2GRAY)

    # --- Stage 1: Stitch A and B ---
    print(f"Stage 1: Stitching {nameA} and {nameB}...")
    good_matches_AB, kpA, kpB = get_sift_matches(grayA, grayB)
    
    if len(good_matches_AB) < 8:
        print(f"Error: Not enough matches for {nameA} -> {nameB}. Skipping this order.")
        return None
    
    pts_src_AB = np.float32([kpA[m.queryIdx].pt for m in good_matches_AB])
    pts_dst_AB = np.float32([kpB[m.trainIdx].pt for m in good_matches_AB])
    idx_AB = np.random.choice(len(pts_src_AB), 8, replace=False)
    H_A_to_B = compute_homography(pts_src_AB[idx_AB], pts_dst_AB[idx_AB])
    
    mosaic_AB = create_mosaic(imgA, imgB, H_A_to_B)
    stage1_path = os.path.join(output_dir, f"bonus_01_mosaic_{nameA}_{nameB}.jpg")
    cv2.imwrite(stage1_path, mosaic_AB)
    print(f"Saved intermediate mosaic to: {stage1_path}")

    # --- Stage 2: Stitch (A+B) and C ---
    print(f"Stage 2: Stitching (Mosaic {nameA}+{nameB}) and {nameC}...")
    gray_mosaic_AB = cv2.cvtColor(mosaic_AB, cv2.COLOR_BGR2GRAY)
    good_matches_mC, kp_m, kpC = get_sift_matches(gray_mosaic_AB, grayC)
    
    if len(good_matches_mC) < 8:
        print(f"Error: Not enough matches for (Mosaic {nameA}+{nameB}) -> {nameC}. Skipping stage 2.")
        return None

    pts_src_mC = np.float32([kp_m[m.queryIdx].pt for m in good_matches_mC])
    pts_dst_mC = np.float32([kpC[m.trainIdx].pt for m in good_matches_mC])
    idx_mC = np.random.choice(len(pts_src_mC), 8, replace=False)
    H_m_to_C = compute_homography(pts_src_mC[idx_mC], pts_dst_mC[idx_mC])
    
    final_mosaic = create_mosaic(mosaic_AB, imgC, H_m_to_C)
    final_path = os.path.join(output_dir, f"bonus_02_final_{nameA}_{nameB}_{nameC}.jpg")
    cv2.imwrite(final_path, final_mosaic)
    print(f"Saved final mosaic to: {final_path}")
    
    return final_mosaic

# =====================================
# Main Execution: Run all 3 orders
# =====================================
output_dir = "output_all_orders"
os.makedirs(output_dir, exist_ok=True)

image1_path = '001.jpg'
image2_path = '002.jpg'
image3_path = '003.jpg'  

img1 = cv2.imread(image1_path)
img2 = cv2.imread(image2_path)
img3 = cv2.imread(image3_path)

if img1 is None or img2 is None:
    print(f"Error: Could not load '{image1_path}' or '{image2_path}'.")
    exit()
if img3 is None:
    print(f"Error: Could not load '{image3_path}'.")
    print("Please add a third image with this name to run the bonus part.")
    exit()

print("All images loaded successfully.")

results = []
titles = []

# --- Order 1: (1+2) + 3 ---
np.random.seed(42) # Reset seed for consistent results
mosaic_12_3 = stitch_three_images(img1, img2, img3, "img1", "img2", "img3", output_dir)
if mosaic_12_3 is not None:
    results.append(mosaic_12_3)
    titles.append("(Img1 + Img2) + Img3")

# --- Order 2: (1+3) + 2 ---
np.random.seed(42) # Reset seed
mosaic_13_2 = stitch_three_images(img1, img3, img2, "img1", "img3", "img2", output_dir)
if mosaic_13_2 is not None:
    results.append(mosaic_13_2)
    titles.append("(Img1 + Img3) + Img2")

# --- Order 3: (2+3) + 1 ---
np.random.seed(42) # Reset seed
mosaic_23_1 = stitch_three_images(img2, img3, img1, "img2", "img3", "img1", output_dir)
if mosaic_23_1 is not None:
    results.append(mosaic_23_1)
    titles.append("(Img2 + Img3) + Img1")
    
print("\nAll stitching orders processed.")

# =====================================
# Display Final Results
# =====================================
if not results:
    print("No mosaics were successfully generated.")
else:
    n_results = len(results)
    # Create a plot with 1 row and n_results columns
    plt.figure(figsize=(n_results * 8, 8))
    
    for i, (mosaic, title) in enumerate(zip(results, titles)):
        plt.subplot(1, n_results, i + 1)
        plt.imshow(cv2.cvtColor(mosaic, cv2.COLOR_BGR2RGB))
        plt.title(f"Final Mosaic: {title}")
        plt.axis("off")
    
    plt.tight_layout()
    plt.show()

print("\nBonus part complete.")