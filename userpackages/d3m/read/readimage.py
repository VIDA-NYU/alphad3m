from __future__ import division

from vistrails.core.modules.vistrails_module import Module

from d3m_ta2_vistrails.common import read_image_file


class ReadImage(Module):
    """ Read the raw image data files and output an array representation
    """
    _input_ports = [('image_list', '(org.vistrails.vistrails.basic:List)')]
    _output_ports = [('image_array', '(org.vistrails.vistrails.basic:List)')]

        
    def compute(self):
        image_list = self.get_input('image_list')

        image_arrays = []
        
        if not image_list is None:        
            for image in image_list:
                out = read_image_file(image)
                image_arrays.append(out['image_array'][0])
            
        self.set_output('image_array', image_arrays)
        
_modules = [ReadImage]
