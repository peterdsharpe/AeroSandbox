import warnings
import copy
import aerosandbox.numpy as np

class Material:
    
    # A short list of properties, more can be added
    
    # Physical Properties
    density                    = None  # kg/m^3
    
    # Mechanical Properties
    tensile_strength_ultimate  = None  # Pa
    tensile_strength_yield     = None  # Pa
    elongation_at_break        = None  # Percentage
    modulus_of_elasticity      = None
    bulk_modulus               = None
    poissons_ratio              = None
    shear_modulus              = None
    
    # Electrical Properties
    electrical_conductivity     = None
    
    # Thermal Properties
    cte_linear                 = None
    specific_heat_capacity     = None
    thermal_conductivity       = None
    
    # Extras
    isotropic                  = None

    # add redirects for easy access? Maybe a bad idea
    _aliases = {
        'density': [
            'rho',
        ],
        'tensile_strength_yield': [
            'yield_strength',
            'yield_stress',
        ],
        'tensile_strength_Ultimate':[
            'ultimate_strength',
            'tensile_strength',  # Not sure I like this, but typical?
        ],
        'elongation_at_break': [
            'elongation',
        ],
        'poissons_ratio': [
            'poisson_ratio'
        ],
        'modulus_of_elasticity':[
            'elastic_modulus',
            'E',
        ],
        'electrical_conductivity': [
            'elastic_modulus'
            'conductivity',
        ],
        'shear_modulus': [
            'modulus_of_rigidity',
            'G',
        ],
        'cte_linear': [
            'coefficient_of_thermal_expansion',
        ],
        'specific_heat_capacity': [
            'specific_heat',
        ]
        }
    
    def _set_aliases(self):
        for key in self._aliases.keys():
            #redirect aliases to original
            
            def make_getter(key):
                def getter(self):
                    # print('getting ' + key)
                    return getattr(self, key)
                return getter
            
            def make_setter(key):
                def setter(self, value):
                    # print('setting ' + key)
                    setattr(self, key, value)
                return setter
            
            for alias in self._aliases[key]:
                prop = property(fget=make_getter(key),
                                fset=make_setter(key))
                
                # Add it to the list of properties
                setattr(self.__class__, alias, prop)      
            
    
    def __init__(self, **kwargs):
        
        self._set_aliases()
        
        # Just stuff  everything into vars for now
        for key in kwargs.keys():
            setattr(self, key, kwargs[key])
      

# %% Web Scraping
      
# A class to import directly from MatMatch? Not 100% reliable, but the lazy way
#class MatMatch_Material(Material):
#    
#    def __init__(self, 
#                 url: str,
#                 force_value: str = None  # min/max
#                 ):
#        self.force_value = force_value
#        
#        warnings.warn('This is a web scraping interface.'+\
#                      '\nAs it depends on an external resource,'+\
#                      ' functionality is not guaranteed.'+\
#                      '\nUse at your own risk.')
#        
#        import requests
#        import json
#        try:
#            from bs4 import BeautifulSoup
#        except ImportError:
#            raise ImportError('Requires BeautifulSoup 4:'+
#                              '/n    pip install beautifulsoup4'+
#                              '/n(not installed by default)')
#        
#        # Get the webpage
#        r = requests.get(url)
#        soup = BeautifulSoup(r.content, 'html.parser')
#        
#        # Hackish interception of the data loading script
#        content = soup.find('script', {'id': '__NEXT_DATA__'}).contents
#        
#        # Get the json data out
#        jsoncontent = json.loads(content[0])
#        
#        # Get the material data
#        sections = jsoncontent['props']['initialProps']['pageProps']['data']['sections']
#        
#        # Get the properties section
#        sections_dict = {x['sectionId']:x for x in sections}
#        
#        # Consolidate all  the sections
#        properties = {}
#        for section in sections_dict['Properties']['sections']:
#            for entry in section['properties']:
#                name = entry['metadata']['name']
#                
#                # make variable format
#                name = name.lower().replace(' ', '_')
#                name = name.replace("'", "")
#                
#                for datapoint in entry['dataPoints']:
#                    if datapoint['metadata']['unitsSystem'] == 'metric':
#                        value_metric = datapoint['value']
#                        units = datapoint['metadata']['unit']
#                    
#                properties.update({name:{'value': value_metric,
#                                         'units': units}})
#                
#        # TODO: Convert to SI (might require pint?)
#        properties = self._convert_to_SI(properties)
#        self.p1 = properties
#        
#        # TODO: Figure out min max
#        properties = self._check_min_max(properties)
#        self.p2 = properties
#        
#        kwargs = {key: properties[key]
#                  for key in properties}
#        
#        super().__init__(**kwargs)
#        
#    @staticmethod
#    def _convert_to_SI(properties):
#        
#        # This can very likely be done much better
#        # Factors are the mul needed to get to SI, same for offsets
#        known_units = {
#            '°C':       {'offset': 272.15},
#            'Ω·m':      {},
#            'W/(m·K)':  {},
#            'J/(kg·K)': {},
#            '1/K':      {},
#            'MPa':      {'factor': 10E6},
#            'GPa':      {'factor': 10E9},
#            'kPa':      {'factor': 10E3},
#            'Pa':       {},
#            '%':        {'factor': 1/100},
#            '[-]':      {},
#            'g/cm³':    {'factor': 10E3},
#            'kg/m³':    {}
#            }
#        
#        props_out = copy.deepcopy(properties)
#        for key in properties.keys():
#            values = properties[key]['value']
#            units = properties[key]['units']
#            
#            if units not in known_units.keys():
#                warnings.warn('Unknown units: ' + units + '\n'\
#                              'No known conversion, these may not be SI')
#                val = ''
#                while val.lower() != 'y' and val.lower() != 'n':
#                    val = input('Include without converting? (y/n) ')
#                    
#                if val.lower() == 'n':
#                    print('Skipping ' + key)
#                    props_out.pop(key)
#                    continue
#                else:
#                    props_out[key] = {'min': values['min'],
#                                       'max': values['max']}
#                    continue
#            
#            mini = values['min'] * known_units[units].get('factor', 1) + \
#                known_units[units].get('offset', 0)
#            maxi = values['max'] * known_units[units].get('factor', 1) + \
#                known_units[units].get('offset', 0)
#                
#            # Let's drop temp for now
#            props_out[key] = {'min': mini, 'max': maxi}
#            
#        return props_out
#    
#    def _check_min_max(self, properties):
#        
#        for key in properties.keys():
#            val = properties[key]
#            
#            if type(self.force_value) == str:
#                force_value = self.force_value
#            elif type(self.force_value) == dict:
#                force_value = self.force_value[key]
#            
#            if force_value == 'max':
#                properties[key] = val['max']
#                continue
#            elif force_value == 'min':
#                properties[key] = val['min']
#                continue
#            elif val['max'] == val['min']:
#                properties[key] = val['max']
#                continue
#            else:
#                warnings.warn('Min/max mismatch for: ' + key + '\n'\
#                              'Please specify value, or force_values in init.')
#                str_in = ''
#                
#                while str_in.lower() != 'min' \
#                    and str_in.lower() != 'max' \
#                    and not np.char.isnumeric(str_in):
#                    str_in = input('Specify value (min/max/float) ')
#                    
#                if np.char.isnumeric(str_in):
#                    properties[key] = float(str_in)
#                    continue
#                if str_in.lower() == 'min':
#                    properties[key] = val['min']
#                    continue
#                if str_in.lower() == 'max':
#                    properties[key] = val['max']
#                    continue
#                else:
#                    input('waddup')
#            
#        return properties
#            
#            
#class AISI_1006_Steel_Cold_Drawn(MatMatch_Material):
#    
#    def __init__(self):
#        url = 'https://matmatch.com/materials/mitf545-aisi-1006-cold-drawn'
#        
#        super().__init__(url, force_value='min')
        