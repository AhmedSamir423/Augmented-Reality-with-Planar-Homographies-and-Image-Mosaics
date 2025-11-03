import cv2
import numpy as np


COVER_PATH       = 'cv_cover.jpg'      
BOOK_VIDEO_PATH  = 'book.mov'          
KONGFU_PANDA    = 'ar_source.mov'    
OUTPUT_VIDEO     = 'ar_output.mov'     

# Output images 
MATCHES_IMG        = 'matches.jpg'          
VERIFY_H_IMG       = 'verify_homography.jpg'
CORNERS_IMG        = 'book_corners.jpg'     
FIRST_OVERLAY_IMG  = 'first_overlay.jpg'    


def get_sift_matches(img1, img2):
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    bf = cv2.BFMatcher()
    knn_matches = bf.knnMatch(des1, des2, k=2)          

    good = []
    for m, n in knn_matches:
        if m.distance < 0.75 * n.distance:             
            good.append(m)
    return good, kp1, kp2


def compute_homography_ransac(pts_src, pts_dst):
   
    if len(pts_src) < 4:
        return None, None
    H, mask = cv2.findHomography(pts_src, pts_dst,
                                method=cv2.RANSAC,
                                ransacReprojThreshold=5.0)
    return H, mask


def crop_center_to_aspect(frame, target_aspect, target_w, target_h):
    
    h, w = frame.shape[:2]

    #HARD-CROP BLACK BARS
    # We crop to remove ~15% from top and bottom
    crop_ratio = 0.15  
    crop_h = int(h * crop_ratio)
    content = frame[crop_h:h-crop_h, :]  

    #Now apply center crop to match book aspect
    ch, cw = content.shape[:2]
    desired_w = int(ch * target_aspect)

    if desired_w > cw:
        desired_w = cw
        desired_h = int(desired_w / target_aspect)
    else:
        desired_h = ch

    start_x = (cw - desired_w) // 2
    start_y = (ch - desired_h) // 2
    cropped = content[start_y:start_y + desired_h, start_x:start_x + desired_w]

    # Resize to exact cover size
    resized = cv2.resize(cropped, (target_w, target_h))
    return resized


def warp_and_overlay(book_frame, ar_cropped, H):
    
    fh, fw = book_frame.shape[:2]

    # warp the AR image
    warped_ar = cv2.warpPerspective(ar_cropped, H, (fw, fh))

    # create a white mask the size of the cover and warp it → book region
    mask = np.full((ar_cropped.shape[0], ar_cropped.shape[1]), 255, dtype=np.uint8)
    warped_mask = cv2.warpPerspective(mask, H, (fw, fh))

    # overlay
    result = book_frame.copy()
    region = warped_mask > 127                     # threshold for safety
    result[region] = warped_ar[region]
    return result



#-----------------------------------------------------------------------------------------

    
cover = cv2.imread(COVER_PATH)
if cover is None:
    raise FileNotFoundError("cv_cover.jpg not found")
book_h, book_w = cover.shape[:2]
book_aspect = book_w / book_h
print(f"Cover loaded: {book_w}×{book_h}  aspect={book_aspect:.3f}")


cap_book = cv2.VideoCapture(BOOK_VIDEO_PATH)
cap_ar   = cv2.VideoCapture(KONGFU_PANDA)
if not cap_book.isOpened() or not cap_ar.isOpened():
    raise RuntimeError("Cannot open one of the videos")


ret, first_book = cap_book.read()
if not ret:
    raise RuntimeError("book.mov is empty")
print("1.1 – Computing SIFT matches on first frame …")
good_matches, kp_cover, kp_first = get_sift_matches(cover, first_book)


best_50 = sorted(good_matches, key=lambda m: m.distance)[:50]
img_matches = cv2.drawMatches(cover, kp_cover,
                                first_book, kp_first,
                                best_50, None,
                                flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
cv2.imwrite(MATCHES_IMG, img_matches)
print(f"   → {len(good_matches)} good matches, saved {MATCHES_IMG}")

#RANSAC HOMOGRAPHY
pts_cover_all = np.float32([kp_cover[m.queryIdx].pt for m in good_matches])
pts_book_all  = np.float32([kp_first[m.trainIdx].pt for m in good_matches])

H, mask = compute_homography_ransac(pts_cover_all, pts_book_all)
if H is None:
    raise RuntimeError("RANSAC failed on first frame")
inliers = mask.ravel() == 1
print(f"   → Homography with {inliers.sum()}/{len(mask)} inliers")

# projected green, actual red
verify = first_book.copy()
for src, dst in zip(pts_cover_all[inliers], pts_book_all[inliers]):
    p_proj = H @ np.array([src[0], src[1], 1.0])
    p_proj = (p_proj[:2] / p_proj[2]).astype(int)
    cv2.circle(verify, tuple(p_proj), 8, (0, 255, 0), 2)   # projected
    cv2.circle(verify, tuple(dst.astype(int)), 8, (0, 0, 255), 2)  # actual
cv2.imwrite(VERIFY_H_IMG, verify)
print(f"   → Verification saved: {VERIFY_H_IMG}")

# GET THE CORNERS OF THE BOOK IN THE FIRST FRAME
cover_corners = np.float32([[0, 0], [book_w, 0], [book_w, book_h], [0, book_h]])
corners_in_book = []
for pt in cover_corners:
    p = H @ np.append(pt, 1.0)
    corners_in_book.append((p[:2] / p[2]).astype(int))
corners_in_book = np.array([corners_in_book])   

corners_vis = first_book.copy()
cv2.polylines(corners_vis, corners_in_book, isClosed=True,
            color=(0, 255, 0), thickness=4)
cv2.imwrite(CORNERS_IMG, corners_vis)
print(f"   → Book corners saved: {CORNERS_IMG}")

#First AR frame overlay 
ret, first_ar = cap_ar.read()
if not ret:
    raise RuntimeError("ar_source.mov is empty")
first_ar_cropped = crop_center_to_aspect(first_ar, book_aspect, book_w, book_h)

first_overlay = warp_and_overlay(first_book, first_ar_cropped, H)
cv2.imwrite(FIRST_OVERLAY_IMG, first_overlay)
print(f"   → First overlay saved: {FIRST_OVERLAY_IMG}")

# Full video 
print("\n1.6 – Building full AR video …")
cap_book.set(cv2.CAP_PROP_POS_FRAMES, 0)   
cap_ar.set(cv2.CAP_PROP_POS_FRAMES, 0)

fps   = cap_book.get(cv2.CAP_PROP_FPS)
w_out = int(cap_book.get(cv2.CAP_PROP_FRAME_WIDTH))
h_out = int(cap_book.get(cv2.CAP_PROP_FRAME_HEIGHT))
n_book = int(cap_book.get(cv2.CAP_PROP_FRAME_COUNT))
n_ar   = int(cap_ar.get(cv2.CAP_PROP_FRAME_COUNT))
n_frames = min(n_book, n_ar)

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
writer = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, fps, (w_out, h_out))

for idx in range(n_frames):
    ret_b, book_f = cap_book.read()
    ret_a, ar_f   = cap_ar.read()
    if not (ret_b and ret_a):
        break

    # per-frame homography (cover → current book frame)
    matches, kp_c, kp_b = get_sift_matches(cover, book_f)
    if len(matches) < 4:
        writer.write(book_f)                 # nothing to overlay
        continue

    src_pts = np.float32([kp_c[m.queryIdx].pt for m in matches])
    dst_pts = np.float32([kp_b[m.trainIdx].pt for m in matches])

    H_frame, mask_frame = compute_homography_ransac(src_pts, dst_pts)
    if H_frame is None or mask_frame.sum() < 10:
        writer.write(book_f)
        continue

    # crop current AR frame 
    ar_crop = crop_center_to_aspect(ar_f, book_aspect, book_w, book_h)

    # warp+overlay
    out_frame = warp_and_overlay(book_f, ar_crop, H_frame)
    writer.write(out_frame)

    if (idx + 1) % 30 == 0:
        print(f"   processed {idx+1}/{n_frames} frames")


cap_book.release()
cap_ar.release()
writer.release()
print("Done")
