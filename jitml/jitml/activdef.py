from dataclasses import dataclass, field
from torch       import nn


@dataclass
class Argof:
    @dataclass
    class ReluFam:
        relufunc: dict = field(default_factory=(dict))
        leakrelu: dict = field(default_factory=(dict))
        pararelu: dict = field(default_factory=(dict))
        randrelu: dict = field(default_factory=(dict))
        capprelu: dict = field(default_factory=(dict))
        elunfunc: dict = field(default_factory=(dict))
        celufunc: dict = field(default_factory=(dict))
        selufunc: dict = field(default_factory=(dict))

    @dataclass
    class SmooFam:
        gelufunc: dict = field(default_factory=(dict))
        silufunc: dict = field(default_factory=(dict))
        mishfunc: dict = field(default_factory=(dict))
        softplus: dict = field(default_factory=(dict))
        softsign: dict = field(default_factory=(dict))

    @dataclass
    class SigmFam:
        sigmfunc: dict = field(default_factory=(dict))
        tanhfunc: dict = field(default_factory=(dict))
        hsigfunc: dict = field(default_factory=(dict))
        htanfunc: dict = field(default_factory=(dict))
        logsigmo: dict = field(default_factory=(dict))

    @dataclass
    class TresFam:
        threshol: dict = field(default_factory=(dict))
        softshrk: dict = field(default_factory=(dict))
        hardshrk: dict = field(default_factory=(dict))
        tanhshrk: dict = field(default_factory=(dict))

    @dataclass
    class GateFam:
        glunfunc: dict = field(default_factory=(dict))
        hswifunc: dict = field(default_factory=(dict))

    @dataclass
    class ClasFam:
        smaxfunc: dict = field(default_factory=(dict))
        sminfunc: dict = field(default_factory=(dict))
        logasmax: dict = field(default_factory=(dict))
        softtwod: dict = field(default_factory=(dict))

    @dataclass
    class SameFam:
        identity: dict = field(default_factory=(dict))

    relufam: ReluFam = field(default_factory=(ReluFam))
    smoofam: SmooFam = field(default_factory=(SmooFam))
    sigmfam: SigmFam = field(default_factory=(SigmFam))
    tresfam: TresFam = field(default_factory=(TresFam))
    gatefam: GateFam = field(default_factory=(GateFam))
    clasfam: ClasFam = field(default_factory=(ClasFam))
    samefam: SameFam = field(default_factory=(SameFam))


class Activations:
    def __init__(self, argof: Argof = Argof()) -> None:
        self.argof = argof
        self.dex   = {
            "relufunc": (argof.relufam.relufunc, nn.ReLU       ),
            "leakrelu": (argof.relufam.leakrelu, nn.LeakyReLU  ),
            "pararelu": (argof.relufam.pararelu, nn.PReLU      ),
            "randrelu": (argof.relufam.randrelu, nn.RReLU      ),
            "capprelu": (argof.relufam.capprelu, nn.ReLU6      ),
            "elunfunc": (argof.relufam.elunfunc, nn.ELU        ),
            "celufunc": (argof.relufam.celufunc, nn.CELU       ),
            "selufunc": (argof.relufam.selufunc, nn.SELU       ),

            "gelufunc": (argof.smoofam.gelufunc ,nn.GELU       ),
            "silufunc": (argof.smoofam.silufunc ,nn.SiLU       ),
            "mishfunc": (argof.smoofam.mishfunc ,nn.Mish       ),
            "softplus": (argof.smoofam.softplus ,nn.Softplus   ),
            "softsign": (argof.smoofam.softsign ,nn.Softsign   ),

            "sigmfunc": (argof.sigmfam.sigmfunc, nn.Sigmoid    ),
            "tanhfunc": (argof.sigmfam.tanhfunc, nn.Tanh       ),
            "hsigfunc": (argof.sigmfam.hsigfunc, nn.Hardsigmoid),
            "htanfunc": (argof.sigmfam.htanfunc, nn.Hardtanh   ),
            "logsigmo": (argof.sigmfam.logsigmo, nn.LogSigmoid ),

            "threshol": (argof.tresfam.threshol, nn.Threshold  ),
            "softshrk": (argof.tresfam.softshrk, nn.Softshrink ),
            "hardshrk": (argof.tresfam.hardshrk, nn.Hardshrink ),
            "tanhshrk": (argof.tresfam.tanhshrk, nn.Tanhshrink ),

            "glunfunc": (argof.gatefam.glunfunc, nn.GLU        ),
            "hswifunc": (argof.gatefam.hswifunc, nn.Hardswish  ),

            "smaxfunc": (argof.clasfam.smaxfunc, nn.Softmax    ),
            "sminfunc": (argof.clasfam.sminfunc, nn.Softmin    ),
            "logasmax": (argof.clasfam.logasmax, nn.LogSoftmax ),
            "softtwod": (argof.clasfam.softtwod, nn.Softmax2d  ),

            "identity": (argof.samefam.identity, nn.Identity   ),
        }

    def get(self, name: str) -> nn.Module:
        if name not in self.dex:
            raise
        argof, myclass = self.dex[name]
        return myclass(**argof)