from .Corruptor import Corruptor
from .DataBend import DataBend
from .dATAmOSH import dATAmOSH
from .FrequencyModulation import FrequencyModulation
from .GlitchIT import GlitchIT
from .interference import interference
from .interferenceV2 import interferenceV2
from .LineScreen import LineScreen
from .LuminousFlow import LuminousFlow
from .GlitchArtNode import GlitchArtNode
from .Peaks import Peaks
from .PixelFloat import PixelFloat
from .PixelRedistribution import PixelRedistribution
from .Rekked import Rekked
from .TvGlitch import TvGlitch
from .VaporWave import VaporWave
from .VideoModulation import VideoModulation

NODE_CLASS_MAPPINGS = {
    "Corruptor": Corruptor,
    "DataBend": DataBend,
    "dATAmOSH": dATAmOSH,
    "FrequencyModulation": FrequencyModulation,
    "GlitchIT": GlitchIT,
    "interference": interference,
    "interferenceV2": interferenceV2,
    "LineScreen": LineScreen,
    "LuminousFlow": LuminousFlow,
    "GlitchArtNode": GlitchArtNode,
    "Peaks": Peaks,
    "PixelFloat": PixelFloat,
    "PixelRedistribution": PixelRedistribution,
    "Rekked": Rekked,
    "TvGlitch": TvGlitch,
    "VaporWave": VaporWave,
    "VideoModulation": VideoModulation,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Corruptor": "Corruptor | GlitchNodes",
    "DataBend": "DataBend | PXLPSHR",
    "dATAmOSH": "░d░A░T░A░m░O░S░H░ | GlitchNodes",
    "FrequencyModulation": "Frequency Modulation | GlitchNodes",
    "GlitchIT": "GlitchIT | GlitchNodes",
    "interference": "interference | GlitchNodes",
    "interferenceV2": "interferenceV2 WIP | GlitchNodes",
    "LineScreen": "LineScreen | GlitchNodes",
    "LuminousFlow": "LuminousFlow | GlitchNodes",
    "GlitchArtNode": "GlitchArtNode | GlitchNodes",
    "Peaks": "Peaks | GlitchNodes",
    "PixelFloat": "PixelFloat | GlitchNodes",
    "PixelRedistribution": "PixelRedistribution WIP | PXLGlitchNodesPSHR",
    "Rekked": "Rekked WIP | GlitchNodes",
    "TvGlitch": "TV Glitch | GlitchNodes",
    "VaporWave": "VaporWave | GlitchNodes",
    "VideoModulation": "VideoModulation | GlitchNodes",
}


ascii_art = """

                      :::!~!!!!!:.
                  .xUHWH!! !!?M88WHX:.
                .X*#M@$!!  !X!M$$$$$$WWx:.
               :!!!!!!?H! :!$!$$$$$$$$$$8X:
              !!~  ~:~!! :~!$!#$$$$$$$$$$8X:
             :!~::!H!<   ~.U$X!?R$$$$$$$$MM!
             ~!~!!!!~~ .:XW$$$U!!?$$$$$$RMM!
               !:~~~ .:!M"T#$$$$WX??#MRRMMM!
               ~?WuxiW*`   `"#$$$$8!!!!??!!!
             :X- M$$$$       `"T#$T~!8$WUXU~
            :%`  ~#$$$m:        ~!~ ?$$$$$$
          :!`.-   ~T$$$$8xx.  .xWW- ~""##*"
.....   -~~:<` !    ~?T#$$@@W@*?$$      /`
W$@@M!!! .!~~ !!     .:XUW$W!~ `"~:    :
#"~~`.:x%`!!  !H:   !WM$$$$Ti.: .!WUn+!`
:::~:!!`:X~ .: ?H.!u "$$$B$$$!W:U!T$$M~
.~~   :X@!.-~   ?@WTWo("*$$$W$TH$! `
Wi.~!X$?!-~    : ?$$$B$Wu("**$RM!
$R@i.~~ !     :   ~$$$$$B$$en:``
?MXT@Wx.~    :     ~"##*$$$$M~                                                                   
▒█▀▀█ ▀▄▒▄▀ ▒█░░░ ▒█▀▀█ ▒█▀▀▀█ ▒█░▒█ ▒█▀▀█ 
▒█▄▄█ ░▒█░░ ▒█░░░ ▒█▄▄█ ░▀▀▀▄▄ ▒█▀▀█ ▒█▄▄▀ 
▒█░░░ ▄▀▒▀▄ ▒█▄▄█ ▒█░░░ ▒█▄▄▄█ ▒█░▒█ ▒█░▒█
~~~~~~~~~ G L I T C H N O D E S ~~~~~~~~~
"""
print(ascii_art)

import time

time.sleep(1)
