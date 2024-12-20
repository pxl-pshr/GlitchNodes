from .Corruptor import Corruptor
from .DataBend import DataBend
from .FrequencyModulation import FrequencyModulation
from .GlitchIT import GlitchIT
from .interference import interference
from .interferenceV2 import interferenceV2
from .LineScreen import LineScreen
from .LuminousFlow import LuminousFlow
from .PixelFloat import PixelFloat
from .PixelRedistribution import PixelRedistribution
from .Rekked import Rekked
from .Scanz import Scanz
from .TvGlitch import TvGlitch
from .VaporWave import VaporWave
from .VHSonAcid import VHSonAcid
from .VideoModulation import VideoModulation

NODE_CLASS_MAPPINGS = {
    "Corruptor": Corruptor,
    "DataBend": DataBend,
    "FrequencyModulation": FrequencyModulation,
    "GlitchIT": GlitchIT,
    "interference": interference,
    "interferenceV2": interferenceV2,
    "LineScreen": LineScreen,
    "LuminousFlow": LuminousFlow,
    "PixelFloat": PixelFloat,
    "PixelRedistribution": PixelRedistribution,
    "Rekked": Rekked,
    "Scanz": Scanz,
    "TvGlitch": TvGlitch,
    "VaporWave": VaporWave,
    "VHSonAcid": VHSonAcid,
    "VideoModulation": VideoModulation,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Corruptor": "Corruptor | GlitchNodes",
    "DataBend": "DataBend | PXLPSHR",
    "FrequencyModulation": "Frequency Modulation | GlitchNodes",
    "GlitchIT": "GlitchIT | GlitchNodes",
    "interference": "interference | GlitchNodes",
    "interferenceV2": "interferenceV2 WIP | GlitchNodes",
    "LineScreen": "LineScreen | GlitchNodes",
    "LuminousFlow": "LuminousFlow | GlitchNodes",
    "PixelFloat": "PixelFloat | GlitchNodes",
    "PixelRedistribution": "PixelRedistribution WIP | PXLGlitchNodesPSHR",
    "Rekked": "Rekked WIP | GlitchNodes",
    "Scanz": "Scanz | GlitchNodes",
    "TvGlitch": "TV Glitch | GlitchNodes",
    "VaporWave": "VaporWave | GlitchNodes",
    "VHSonAcid": "VHSonAcid | GlitchNodes",
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
