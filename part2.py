import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# =====================================
# Setup and Image Loading
# =====================================
os.makedirs("output", exist_ok=True)  # Create output folder if not exists

image1_path = 'pano_image1.jpg'
image2_path = 'pano_image2.jpg'  # fixed (was same image before)

img1 = cv2.imread(image1_path)
img2 = cv2.imread(image2_path)

gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

# =====================================
# 1. SIFT Feature Detection and Matching
# =====================================
def get_sift_matches(img1_gray, img2_gray, ratio_thresh=0.75, max_matches=50):
    """Detect and match SIFT keypoints using Lowe’s ratio test, returning top matches."""
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(img1_gray, None)
    kp2, des2 = sift.detectAndCompute(img2_gray, None)

    # Brute-force matcher with KNN
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
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


# =====================================
# 2. Compute Homography using Manual DLT
# =====================================
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


# =====================================
# 3. Feature Matching and Homography Estimation
# =====================================
good_matches, kp1, kp2 = get_sift_matches(gray1, gray2)
print(f"Number of good matches: {len(good_matches)}")

# Visualize top matches
img_matches = cv2.drawMatches(
    img1, kp1, img2, kp2, good_matches, None,
    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
)
cv2.imwrite("01_sift_matches.jpg", img_matches)

plt.figure(figsize=(16, 8))
plt.imshow(cv2.cvtColor(img_matches, cv2.COLOR_BGR2RGB))
plt.title(f"Top {len(good_matches)} SIFT Matches")
plt.axis("off")
plt.show()

# Extract matched keypoints
pts_src = np.float32([kp1[m.queryIdx].pt for m in good_matches])
pts_dst = np.float32([kp2[m.trainIdx].pt for m in good_matches])

# Use 8 random correspondences to compute homography
np.random.seed(42)
idx = np.random.choice(len(pts_src), 8, replace=False)
H_manual = compute_homography(pts_src[idx], pts_dst[idx])
print("Computed Homography (manual DLT):\n", H_manual)


# =====================================
# 4. Homography Verification
# =====================================
def project_points(H, pts):
    """Project points using homography."""
    pts_hom = np.hstack([pts, np.ones((pts.shape[0], 1))])
    projected = (H @ pts_hom.T).T
    projected /= projected[:, [2]]
    return projected[:, :2]


subset = np.random.choice(len(pts_src), 10, replace=False)
pts_projected = project_points(H_manual, pts_src[subset])

plt.figure(figsize=(8, 6))
plt.imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))
plt.scatter(pts_dst[subset, 0], pts_dst[subset, 1], c='lime', label='Actual Points')
plt.scatter(pts_projected[:, 0], pts_projected[:, 1], c='red', marker='x', label='Projected Points')
plt.legend()
plt.title("Projection of Source Points onto Destination Image")
plt.axis("off")
plt.show()


# =====================================
# 5. Bilinear Interpolation
# =====================================
def bilinear_interpolate(img, x, y):
    """Perform bilinear interpolation for a single channel image."""
    h, w = img.shape
    x0 = np.floor(x).astype(int)
    x1 = x0 + 1
    y0 = np.floor(y).astype(int)
    y1 = y0 + 1

    x0 = np.clip(x0, 0, w - 1)
    x1 = np.clip(x1, 0, w - 1)
    y0 = np.clip(y0, 0, h - 1)
    y1 = np.clip(y1, 0, h - 1)

    Ia = img[y0, x0]
    Ib = img[y1, x0]
    Ic = img[y0, x1]
    Id = img[y1, x1]

    wa = (x1 - x) * (y1 - y)
    wb = (x1 - x) * (y - y0)
    wc = (x - x0) * (y1 - y)
    wd = (x - x0) * (y - y0)

    return wa * Ia + wb * Ib + wc * Ic + wd * Id


# =====================================
# 6. Image Warping (Inverse Warping)
# =====================================
def warp_image(img, H):
    """Warp img using homography H with inverse warping (bilinear interpolation)."""
    h, w = img.shape[:2]

    # Define corner points of img
    corners = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)
    corners_hom = np.hstack([corners, np.ones((4, 1))])

    # Transform corners using H
    warped_corners = (H @ corners_hom.T).T
    warped_corners /= warped_corners[:, [2]]

    min_x, min_y = np.floor(warped_corners[:, 0].min()), np.floor(warped_corners[:, 1].min())
    max_x, max_y = np.ceil(warped_corners[:, 0].max()), np.ceil(warped_corners[:, 1].max())

    out_w, out_h = int(max_x - min_x), int(max_y - min_y)

    # Generate grid in destination (warped) image
    xx, yy = np.meshgrid(np.arange(out_w), np.arange(out_h))
    dest_pts = np.stack([xx + min_x, yy + min_y, np.ones_like(xx)], axis=-1).reshape(-1, 3)

    # Inverse warp to source
    H_inv = np.linalg.inv(H)
    src_pts = (H_inv @ dest_pts.T).T
    src_pts /= src_pts[:, [2]]

    src_x = src_pts[:, 0]
    src_y = src_pts[:, 1]

    # Create empty warped image
    warped = np.zeros((out_h, out_w, 3), dtype=np.uint8)

    for c in range(3):
        channel = img[:, :, c]
        sampled = bilinear_interpolate(channel, src_x, src_y)
        warped[:, :, c] = sampled.reshape(out_h, out_w)

    return warped, (min_x, min_y)


warped_img, offset = warp_image(img1, H_manual)
cv2.imwrite("02_warped_image.jpg", warped_img)

plt.figure(figsize=(12, 8))
plt.imshow(cv2.cvtColor(warped_img, cv2.COLOR_BGR2RGB))
plt.title("Warped Image (img1 warped into img2’s plane)")
plt.axis("off")
plt.show()


# =====================================
# 7. Mosaic Creation
# =====================================
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
    mosaic[translation[1]:translation[1]+h2, translation[0]:translation[0]+w2] = img2

    return mosaic


mosaic = create_mosaic(img1, img2, H_manual)
cv2.imwrite("03_mosaic.jpg", mosaic)

plt.figure(figsize=(14, 8))
plt.imshow(cv2.cvtColor(mosaic, cv2.COLOR_BGR2RGB))
plt.title("Image Mosaic")
plt.axis("off")
plt.show()
print(" - 01_sift_matches.jpg")
print(" - 02_warped_image.jpg")
print(" - 03_mosaic.jpg")
