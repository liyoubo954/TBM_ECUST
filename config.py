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
        "GZC_PRS1", "KWC_PRS4", "PJB_JNK_PRS2.1", "YQMF_FK_PRS", "ZQD_SS_PRS1", "CLY_YQ_PRS", "YQMF_XLJC_QPRS",
        "DW_PRS4.2", "DW_PRS4.4", "DW_PRS78", "DW_PRS81", "DW_PRS84", "DW_PRS87_2", "DW_PRS89", "DW_PRS91",
        "AYB_PRS1_1", "AYB_PRS1_2", "AYB_PRS1_3", "AYB_PRS1_4", "AYB_PRS5", "AYB_PRS6", "AYB_PRS7", "AYB_PRS8",
        "TJSD", "TJL", "DP_ZJ", "GRD", "DP_SD", "state", "RING"
    ]


