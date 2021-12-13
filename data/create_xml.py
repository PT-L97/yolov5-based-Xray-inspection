import os, sys
import glob
from PIL import Image
 
# non-target img path
src_img_dir = "negative_sample/0"
# the created non-target img.xml path
src_xml_dir = "negative_sample/XML"
 
img_Lists = glob.glob(src_img_dir + '/*.jpg')
 
img_basenames = [] # e.g. 100.jpg
for item in img_Lists:
    img_basenames.append(os.path.basename(item))
 
img_names = [] # e.g. 100
for item in img_basenames:
    temp1, temp2 = os.path.splitext(item)
    img_names.append(temp1)
 
for img in img_names:
    im = Image.open((src_img_dir + '/' + img + '.jpg'))
    width, height = im.size
    # write in xml file
    #os.mknod(src_xml_dir + '/' + img + '.xml')
    xml_file = open((src_xml_dir + '/' + img + '.xml'), 'w')
    xml_file.write('<annotation>\n')
    xml_file.write('    <folder>X_ray</folder>\n')
    xml_file.write('    <filename>' + str(img) + '.jpg' + '</filename>\n')
    xml_file.write('    <path>'+ src_xml_dir + '/' + str(img) + '.jpg' + '</path>\n')
    xml_file.write('    <source>\n')
    xml_file.write('        <database>' + "The X_ray Database" + '</database>\n')
    xml_file.write('    </source>\n')
    xml_file.write('    <size>\n')
    xml_file.write('        <width>' + str(width) + '</width>\n')
    xml_file.write('        <height>' + str(height) + '</height>\n')
    xml_file.write('        <depth>3</depth>\n')
    xml_file.write('    </size>\n')
    xml_file.write('    <segmented>0</segmented>\n')
    xml_file.write('</annotation>')