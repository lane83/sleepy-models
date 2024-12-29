def __init__(self, data_dir: str = "data"):
    """
    Initialize the configuration manager.
    
    Args:
        data_dir (str): Directory to store configuration data
    """
    self.data_dir = data_dir
    os.makedirs(data_dir, exist_ok=True)
    
    self.config_file = os.path.join(data_dir, "config_backups.json")
    self.lock_file = os.path.join(data_dir, "config.lock")
    self.key_file = os.path.join(data_dir, "config.key")
    self.file_lock = FileLock(self.lock_file)
    
    # Load environment variables
    load_dotenv()
    
    # Initialize encryption
    self.encryption_key = self._load_or_create_key()
    self.fernet = Fernet(self.encryption_key)
    
    self.load_configs()

def _load_or_create_key(self) -> bytes:
    """
    Load or create encryption key.
    
    Returns:
        bytes: Encryption key
    """
    try:
        if os.path.exists(self.key_file):
            with open(self.key_file, "rb") as f:
                return f.read()
        else:
            key = Fernet.generate_key()
            with open(self.key_file, "wb") as f:
                f.write(key)
            return key
    except Exception as e:
        print(f"Error handling encryption key: {e}")
        return Fernet.generate_key()

def load_configs(self) -> None:
    """Load configurations from file with proper error handling."""
    with self.file_lock:
        try:
            if os.path.exists(self.config_file):
                with open(self.config_file, 'r') as f:
                    self.configs = json.load(f)
            else:
                self.configs = {
                    "versions": [],
                    "presets": {}
                }
        except Exception as e:
            print(f"Error loading configurations: {e}")
            self.configs = {
                "versions": [],
                "presets": {}
            }

def save_configs(self) -> None:
    """Save configurations with atomic write operations."""
    with self.file_lock:
        try:
            temp_file = f"{self.config_file}.tmp"
            with open(temp_file, 'w') as f:
                json.dump(self.configs, f)
            os.replace(temp_file, self.config_file)
        except Exception as e:
            print(f"Error saving configurations: {e}")
            if os.path.exists(temp_file):
                try:
                    os.remove(temp_file)
                except:
                    pass

def _encrypt_sensitive_data(self, config: Dict) -> Dict:
    """
    Encrypt sensitive configuration data.
    
    Args:
        config (Dict): Configuration to encrypt
        
    Returns:
        Dict: Configuration with encrypted sensitive data
    """
    encrypted_config = config.copy()
    sensitive_fields = ['hf_api_key', 'anthropic_api_key', 'openai_api_key']
    
    for field in sensitive_fields:
        if field in encrypted_config and encrypted_config[field]:
            try:
                encrypted_config[field] = self.fernet.encrypt(
                    encrypted_config[field].encode()
                ).decode()
            except Exception as e:
                print(f"Error encrypting {field}: {e}")
                encrypted_config[field] = ""
    
    return encrypted_config

def _decrypt_sensitive_data(self, config: Dict) -> Dict:
    """
    Decrypt sensitive configuration data.
    
    Args:
        config (Dict): Configuration with encrypted data
        
    Returns:
        Dict: Configuration with decrypted sensitive data
    """
    decrypted_config = config.copy()
    sensitive_fields = ['hf_api_key', 'anthropic_api_key', 'openai_api_key']
    
    for field in sensitive_fields:
        if field in decrypted_config and decrypted_config[field]:
            try:
                decrypted_config[field] = self.fernet.decrypt(
                    decrypted_config[field].encode()
                ).decode()
            except Exception as e:
                print(f"Error decrypting {field}: {e}")
                decrypted_config[field] = ""
    
    return decrypted_config

def get_default_config(self) -> Dict:
    """
    Get default configuration settings.
    
    Returns:
        Dict: Default configuration
    """
    return {
        "local_model": "microsoft/phi-2",
        "cloud_model": "anthropic/claude-3-sonnet",
        "hf_api_key": os.getenv("HUGGINGFACE_API_KEY", ""),
        "anthropic_api_key": os.getenv("ANTHROPIC_API_KEY", ""),
        "openai_api_key": os.getenv("OPENAI_API_KEY", ""),
        "local_model_provider": "huggingface",
        "cloud_model_provider": "anthropic",
        "performance_thresholds": {
            "max_response_time": 5.0,
            "min_token_efficiency": 10.0,
            "max_error_rate": 0.05,
            "max_cost_per_token": 0.0002
        },
        "dream_state_triggers": {
            "token_threshold": 0.8,
            "time_threshold": 3600,
            "error_threshold": 0.1
        }
    }

def save_config_version(self, config: Dict, name: str = None) -> str:
    """
    Save a new configuration version.
    
    Args:
        config (Dict): Configuration to save
        name (str, optional): Name for the configuration
        
    Returns:
        str: Version ID
    """
    try:
        # Encrypt sensitive data
        encrypted_config = self._encrypt_sensitive_data(config)
        
        # Create version
        version = {
            "id": datetime.now().strftime("%Y%m%d_%H%M%S"),
            "timestamp": datetime.now().isoformat(),
            "name": name or f"Config {len(self.configs['versions']) + 1}",
            "config": encrypted_config,
            "checksum": hashlib.sha256(
                json.dumps(encrypted_config, sort_keys=True).encode()
            ).hexdigest()
        }
        
        self.configs["versions"].append(version)
        self.save_configs()
        return version["id"]
    except Exception as e:
        print(f"Error saving config version: {e}")
        return ""

def save_preset(self, config: Dict, name: str) -> bool:
    """
    Save a configuration preset.
    
    Args:
        config (Dict): Configuration to save as preset
        name (str): Name for the preset
        
    Returns:
        bool: Success status
    """
    try:
        encrypted_config = self._encrypt_sensitive_data(config)
        self.configs["presets"][name] = {
            "config": encrypted_config,
            "created": datetime.now().isoformat(),
            "last_modified": datetime.now().isoformat()
        }
        self.save_configs()
        return True
    except Exception as e:
        print(f"Error saving preset: {e}")
        return False

def get_config_version(self, version_id: str) -> Optional[Dict]:
    """
    Get a specific configuration version.
    
    Args:
        version_id (str): ID of the version to retrieve
        
    Returns:
        Optional[Dict]: Configuration or None if not found
    """
    try:
        for version in self.configs["versions"]:
            if version["id"] == version_id:
                return self._decrypt_sensitive_data(version["config"])
        return None
    except Exception as e:
        print(f"Error getting config version: {e}")
        return None

def get_preset(self, name: str) -> Optional[Dict]:
    """
    Get a specific preset configuration.
    
    Args:
        name (str): Name of the preset
        
    Returns:
        Optional[Dict]: Preset configuration or None if not found
    """
    try:
        preset = self.configs["presets"].get(name)
        if preset:
            return self._decrypt_sensitive_data(preset["config"])
        return None
    except Exception as e:
        print(f"Error getting preset: {e}")
        return None

def list_versions(self) -> List[Dict]:
    """
    Get list of available configuration versions.
    
    Returns:
        List[Dict]: List of version metadata
    """
    return [{
        "id": v["id"],
        "name": v["name"],
        "timestamp": v["timestamp"]
    } for v in self.configs["versions"]]

def list_presets(self) -> List[str]:
    """
    Get list of available presets.
    
    Returns:
        List[str]: List of preset names
    """
    return list(self.configs["presets"].keys())

def export_config(self, version_id: str, filepath: str) -> bool:
    """
    Export a configuration version to file.
    
    Args:
        version_id (str): ID of the version to export
        filepath (str): Path to export file
        
    Returns:
        bool: Success status
    """
    try:
        config = self.get_config_version(version_id)
        if config:
            with open(filepath, 'w') as f:
                json.dump(config, f, indent=2)
            return True
        return False
    except Exception as e:
        print(f"Error exporting config: {e}")
        return False

def import_config(self, filepath: str) -> str:
    """
    Import a configuration from file.
    
    Args:
        filepath (str): Path to import file
        
    Returns:
        str: Version ID of imported configuration
    """
    try:
        with open(filepath, 'r') as f:
            config = json.load(f)
        return self.save_config_version(
            config,
            f"Imported {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        )
    except Exception as e:
        print(f"Error importing config: {e}")
        return ""