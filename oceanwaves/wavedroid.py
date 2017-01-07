import numpy as np
import re
import oceanwaves
from datetime import datetime
import pytz

WD_TIME_FORMAT = '%Y-%m-%d %H:%M:%S'

class WaveDroidReader:


    def __init__(self):

        self.reset()
        

    def reset(self):
        
        self.stationary = False
        self.directional = False

        self.version = None
        self.timecoding = None
        self.comments = []

        self.time = []
        self.energy_units = None
        self.frequencies = []
        self.directions = []
        self.energy = []
        

    def __call__(self, fpath):

        self.reset()
        return self.read(fpath)


    def readfile(self, fpath):

        with open(fpath, 'r') as fp:
            self.lines = fp.readlines()
        
        self.n = 0 # line counter
        
        self.parse_header()
        
        self.parse_data()
        
        self.make_aware() # Add timezone to self.time
        
        if any(self.frequencies):
            self.compute_spectrum()
            self.energy_units = 'm^2/Hz'
        else:
            self.energy = self.Hm0[:,None,None,None]
            self.energy_units = 'm'        
        
        
    def to_oceanwaves(self):
        
        loc = [(self.longitude,self.latitude)]
        kwargs = dict(location=loc,
                      location_units='deg',
                      frequency=self.frequencies,
                      frequency_units='Hz',
                      frequency_convention='absolute',
                      energy=self.energy[:,None,:,:],
                      energy_units=self.energy_units,
                      time=self.time,
                      time_units='s',
                      attrs=dict(comments=self.comments))

        if self.directional:
            kwargs.update(dict(direction=self.directions,
                               direction_units='deg',
                               direction_convention='nautical'))

        return oceanwaves.OceanWaves(**kwargs)
        
        
    def parse_header(self):
        
        while not re.match('^[0-9]',self._currentline()):
            line = self._currentline()
            
            if line.startswith('version'):
                self.parse_version()
            elif line.startswith('ID'):
                self.parse_id()                
            elif line.startswith('location'):
                self.parse_location()
            elif line.startswith('latitude'):
                self.parse_lat()                
            elif line.startswith('longitude'):
                self.parse_lon()                
            elif line.startswith('magdec'):
                self.parse_magdec()                
            elif line.startswith('timezone'):
                self.parse_tz()                
            elif line.startswith('frequencies'):
                self.parse_freq()                
            
            self.n += 1
    
    def parse_data(self):
        lines = self._currentlines()
        
        self.Hm0 = []
        self.Tp = []
        self.Dirp = []
        self.Tavg = []
        self.Hmax = []
        self.Tmax = []
        
        if any(self.frequencies):       
            self.Puu = []
            self.th0 = []
            self.m1 = []
            self.m2 = []
            self.n2 = []
            
            self.directional = True
        
        while lines:
            line = lines.pop(0)
            
            vals = re.split(';',line)
            self.time.append(datetime.strptime(vals[0],WD_TIME_FORMAT))
            self.Hm0.append(float(vals[1]))
            self.Tp.append(float(vals[2]))
            self.Dirp.append(float(vals[3]))
            self.Tavg.append(float(vals[4]))
            self.Hmax.append(float(vals[5]))
            self.Tmax.append(float(vals[6]))
            
            if any(self.frequencies):
                self.Puu.append(np.asarray(vals[7].split(','),dtype=float))
                self.th0.append(np.asarray(vals[8].split(','),dtype=float))
                self.m1.append(np.asarray(vals[9].split(','),dtype=float))
                self.m2.append(np.asarray(vals[10].split(','),dtype=float))
                self.n2.append(np.asarray(vals[11].split(','),dtype=float))                
    
    def make_aware(self):
        
        tz = pytz.timezone(self.timezone)
        self.time = map(tz.localize,self.time)

    
    def parse_version(self):
        line = self._currentline()
        
        m = re.search('(?<=version\s=\s)[0-9]*',line)
        self.version = int(m.group())
        
        
    def parse_id(self):
        line = self._currentline()
        
        m = re.search('(?<=ID\s=\s)WD[0-9]*',line)
        self.id = m.group()
        
        self.comments.append('WaveDroid ID = %s' %(self.id))
        
        
    def parse_location(self):
        line = self._currentline()
        
        m = re.search('(?<=location\s=\s).*',line)
        self.location = m.group()

        self.comments.append('WaveDroid location = %s' %(self.location))        
        
        
    def parse_lat(self):
        line = self._currentline()
        
        m = re.search('(?<=latitude\s=\s)[0-9\.-]*',line)
        self.latitude = float(m.group())

        
    def parse_lon(self):
        line = self._currentline()
        
        m = re.search('(?<=longitude\s=\s)[0-9\.-]*',line)
        self.longitude = float(m.group())


    def parse_magdec(self):
        line = self._currentline()
        
        m = re.search('(?<=magdec\s=\s)[0-9\.-]*',line)
        self.magdec = float(m.group())
        

    def parse_tz(self):
        line = self._currentline()
        
        m = re.search('(?<=timezone\s=\s)[A-Za-z0-9/_]*',line)
        self.timezone = m.group()
        
        self.comments.append('Timezone = %s' %(self.timezone))
        

    def parse_freq(self):
        line = self._currentline()
        
        m = re.search('(?<=frequencies\s=\s)[0-9\.,]*',line)
        self.frequencies = np.asarray(m.group().split(','),dtype=float)
    
    
    def compute_spectrum(self,ndir=360):
        if np.remainder(ndir,2) != 0:
            raise ValueError('Number of directional bins should be even!')
        
        spec = np.empty((len(self.time),len(self.frequencies),ndir))
        
        rad = np.linspace(-np.pi,np.pi-2*np.pi/ndir,ndir)
        
        self.th0_cor = map(self.add_magdec,self.th0)
        for i,(p,t,m,mm,nn) in enumerate(zip(self.Puu,self.th0_cor,self.m1,self.m2,self.n2)):
            for j,r in enumerate(rad):
                spec[i,:,j] = 1/np.pi*(.5 + m*np.cos(r-t) + mm*np.cos(2*(r-t)) + nn*np.sin(2*(r-t)))*p
        
        self.energy = np.roll(spec,ndir/2,axis=2) # Shift from -pi->pi to 0->2pi
        
        self.directions = np.linspace(0.,360.-360./ndir,ndir)

    def add_magdec(self,x):
        return x + self.magdec


    def _currentline(self):
        return self.lines[self.n]
        
        
    def _currentlines(self):
        return self.lines[self.n:]
