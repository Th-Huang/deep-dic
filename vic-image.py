import os
from PIL import Image, ImageDraw
img_path = '../DIC-dataset/test/original/t1-00000000_0.tif'
def circle(img_path):
    path_name = os.path.dirname(img_path)
    path_name = os.path.dirname(path_name)
    cir_file_name = 'cropped_1.tif'
    cir_path = path_name + '/cropped/' + cir_file_name
    ima = Image.open(img_path).convert('RGBA')
    size = ima.size
    print("size:", size)

    r2 = min(size[0], size[1])
    if size[0] != size[1]:
        ima = ima.resize((r2, r2), Image.LANCZOS)

    r3 = int(r2 / 2)
    imb = Image.new('RGBA', (r3*2, r3*2), "white")
    pima = ima.load()
    pimb = imb.load()
    r = float(r2/2)

    for i in range(r2):
        for j in range(r2):
            lx = abs(i-r)
            ly = abs(j-r)
            l = (pow(lx, 2) + pow(ly, 2)) ** 0.5
            if l < r3:
                pimb[i-(r-r3),j-(r-r3)] = pima[i, j]

    imb.save(cir_path)
    return cir_path

if __name__ == '__main__':
    img_path = '../DIC-dataset/test/original/t1-00000000_0.tif'
    cir_path = circle(img_path)
    print("Cropped image saved at:", cir_path)