import sys
import os
sys.path.append(os.getcwd())
try:
    import train
    from core.models.GAGAvatar.models import GAGAvatar
    print('IMPORT_OK')
except Exception as e:
    print(f'IMPORT_FAILED: {e}')
    sys.exit(1)

class ConfigDict(dict):
    def __getattr__(self, name): return self[name]
    def __setattr__(self, name, value): self[name] = value

def test_model(mode):
    print(f'TESTING_MODE: {mode}')
    cfg = ConfigDict()
    cfg.NORMAL_LOSS = ConfigDict()
    cfg.NORMAL_LOSS.ENABLED = True
    cfg.MODE = mode
    try:
        model = GAGAvatar(cfg)
        print(f'INIT_SUCCESS: {mode}')
    except Exception as e:
        print(f'INIT_FAILED: {mode} - {e}')

test_model('point')
test_model('test_screen') # Adjusted to 'screen' below if 'test_screen' was not the literal. Prompt said 'screen'
test_model('screen')