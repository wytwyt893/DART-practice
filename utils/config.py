import yaml

def load_config(config_path: str) -> dict: 
    """
    从 YAML 文件中读取实验配置。
    """
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f) # yaml.safe_load() 是 PyYAML 库提供的一个函数，用于从 YAML 格式的字符串或文件中解析数据，并将其转换为 Python 对象（通常是字典）。相比于 yaml.load()，yaml.safe_load() 在解析过程中会限制一些不安全的 YAML 标签，避免执行潜在的恶意代码，因此更推荐使用 safe_load 来加载配置文件。
    return config


