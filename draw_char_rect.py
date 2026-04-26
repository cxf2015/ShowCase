import cv2
image_path = r"D:\Formax\260420\0421\qingxitu\qx_page_0001.tif"
# char_rect_file_path = r'D:\Formax\260420\0421\qingxitu\qx_page_0001.b'
# char_rect_file_path = r'D:\Formax\260420\0421\qingxitu\qx_page_0001.c'
char_rect_file_path = r'D:\Formax\260420\0421\qingxitu\qx_page_0001.d'

image = cv2.imread(image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# 读取字符区域信息 utf-16-le格式读取
# with open(char_rect_file_path, 'r', encoding='utf-16-le') as f:
with open(char_rect_file_path, 'r', encoding='utf-8') as f:
    '''
    '“'(350,349,393,387)0.9850
    '为'(393,349,436,387)0.9850
    '什'(436,349,480,387)0.9850
    '''
    char_rects = []
    for line in f:
        if len(line.strip())<=0:
            continue
        print(line)
        rect = line.strip().split('(')[1].split(')')[0].split(',')
        print(rect)
        rect = [int(i) for i in rect]
        print(rect)
        char_rects.append(rect)
    print(char_rects)
    for rect in char_rects:
        cv2.rectangle(image, rect[:2], rect[2:], (0, 0, 255), 2)
        cv2.line(image, rect[:2], rect[2:], (0, 255, 255), 1)
        cv2.line(image, rect[2:], rect[:2], (0, 255, 255), 1)
        # cv2.putText(image, str(rect), rect[:2], cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        print(rect)
        print(rect[:2])
        print(rect[2:])
        print(rect[:2][0])
    # 显示图片自动缩放
    image = cv2.resize(image, (int(image.shape[1] * 0.5), int(image.shape[0] * 0.5)))

    cv2.imshow('image', image)
    cv2.waitKey(0)