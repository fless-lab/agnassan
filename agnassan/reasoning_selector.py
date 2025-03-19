from agnassan.models import LocalModelInterface
import torch
import logging

class ReasoningSelector:
    def __init__(self, reasoning_engine):
        self.logger = logging.getLogger("agnassan.reasoning_selector")
        self.reasoning_engine = reasoning_engine

    async def select_technique_with_model(self, query, model_interface):
        try:
            # Vérification de la disponibilité de PyTorch
            if not hasattr(torch, 'cuda'):
                raise ImportError("PyTorch n'est pas correctement initialisé")

            # Logique de sélection des techniques de raisonnement
            return ['chain_of_thought']
        except Exception as e:
            self.logger.error(f"Erreur de sélection de technique: {str(e)}", exc_info=True)
            raise