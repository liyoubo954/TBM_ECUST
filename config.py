import os


class Config(object):
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'sf9glGVa20pyM1NtdukZ'
    INFLUXDB_HOST = os.environ.get('INFLUXDB_HOST') or '192.168.211.108'
    INFLUXDB_PORT = int(os.environ.get('INFLUXDB_PORT') or 38086)
    INFLUXDB_USERNAME = os.environ.get('INFLUXDB_USERNAME') or 'admin'
    INFLUXDB_PASSWORD = os.environ.get('INFLUXDB_PASSWORD') or 'FZaStb0cXFuFbehPBM6YHCiuAAX6QIXr'
    INFLUXDB_DATABASE = os.environ.get('INFLUXDB_DATABASE') or 'algorithm'
    INFLUXDB_MEASUREMENT = os.environ.get('INFLUXDB_MEASUREMENT') or 'tsjy_dz1360_riskwarning'
    INFLUXDB_TIMEOUT = int(os.environ.get('INFLUXDB_TIMEOUT') or 10)

    # 项目字段
    TSJY_FIELDS = [
        "WorkCham.Pres.01", "ExcavCham.Pres.04", "SlurryPump.P2.1.MudIn.Pres", "MB.InSeal.Grs.Pres", "MD.TelSeal.Grs.Pres.01", "GOil.OilGasSeal.Pres", "OilGasSeal.LeakDetCham.Pres",
        "ShieldTail.Seal.Rear.Pres.02", "ShieldTail.Seal.Rear.Pres.04", "ShieldTail.Seal.Rear.Pres.06", "ShieldTail.Seal.Rear.Pres.09", "ShieldTail.Seal.Rear.Pres.12", "ShieldTail.Seal.Rear.Pres.15", "ShieldTail.Seal.Rear.Pres.17", "ShieldTail.Seal.Rear.Pres.19",
        "Liquid.Vlv.A.OutPres.01", "Liquid.Vlv.A.OutPres.02", "Liquid.Vlv.A.OutPres.03", "Liquid.Vlv.A.OutPres.04", "Liquid.Vlv.A.OutPres.05", "Liquid.Vlv.A.OutPres.06", "Liquid.Vlv.A.OutPres.07", "Liquid.Vlv.A.OutPres.08",
        "Prop.Spd", "Thrust", "CutterHead.Torque", "GRD", "CutterHead.Spd", "CutterHead.Total.Extr.Pres", "state", "Ring.No"
    ]
