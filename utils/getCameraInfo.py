from xml.dom import minidom
import os


def getCameraInfo(file):
    if os.path.isdir(file):
        file = os.path.join(file, 'ch0.xml')

    if not os.path.exists(file):
        raise ValueError('File %s does not exist.'%(file))

    camera_info = dict()
    xmldoc = minidom.parse(file)

    itemlist = xmldoc.getElementsByTagName('info')
    for s in itemlist:
        camera_info.update(dict(s.attributes.items()))

    itemlist = xmldoc.getElementsByTagName('action')
        for s in itemlist:
            camera_info.update(dict(s.attributes.items()))

    return camera_info
