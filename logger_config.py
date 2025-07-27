"""
Configuração centralizada do sistema de logging para o projeto SIAC.
Fornece loggers estruturados com diferentes níveis e formatação consistente.
"""

import logging
import os
from datetime import datetime
from typing import Optional

class SiacLogger:
    """
    Classe para configuração e gerenciamento centralizado de logs do sistema SIAC.
    """
    
    _loggers = {}
    _log_dir = "logs"
    _initialized = False
    
    @classmethod
    def setup_logging(cls, log_level: str = "INFO", enable_file_logging: bool = True) -> None:
        """
        Configura o sistema de logging global.
        
        Args:
            log_level: Nível de log (DEBUG, INFO, WARNING, ERROR, CRITICAL)
            enable_file_logging: Se True, salva logs em arquivo além do console
        """
        if cls._initialized:
            return
            
        # Criar diretório de logs se necessário
        if enable_file_logging:
            os.makedirs(cls._log_dir, exist_ok=True)
        
        # Configurar formato dos logs
        log_format = '[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s'
        date_format = '%Y-%m-%d %H:%M:%S'
        
        # Configurar handler para console
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(logging.Formatter(log_format, date_format))
        
        # Configurar handler para arquivo (se habilitado)
        handlers = [console_handler]
        if enable_file_logging:
            log_filename = f"siac_{datetime.now().strftime('%Y%m%d')}.log"
            log_filepath = os.path.join(cls._log_dir, log_filename)
            
            file_handler = logging.FileHandler(log_filepath, encoding='utf-8')
            file_handler.setFormatter(logging.Formatter(log_format, date_format))
            handlers.append(file_handler)
        
        # Configurar logging básico
        logging.basicConfig(
            level=getattr(logging, log_level.upper()),
            handlers=handlers,
            format=log_format,
            datefmt=date_format
        )
        
        cls._initialized = True
        
        # Log de inicialização
        root_logger = cls.get_logger("SIAC_SYSTEM")
        root_logger.info("Sistema de logging inicializado")
        root_logger.info(f"Nível de log: {log_level}")
        root_logger.info(f"Log em arquivo: {'Habilitado' if enable_file_logging else 'Desabilitado'}")
    
    @classmethod
    def get_logger(cls, name: str) -> logging.Logger:
        """
        Obtém um logger com nome específico.
        
        Args:
            name: Nome do logger (ex: 'DETECTOR', 'STATE_MANAGER', etc.)
            
        Returns:
            Logger configurado
        """
        if name not in cls._loggers:
            logger = logging.getLogger(name)
            cls._loggers[name] = logger
        
        return cls._loggers[name]
    
    @classmethod
    def log_detection_stats(cls, logger: logging.Logger, roi_count: int, 
                          items_count: int, divisors_count: int) -> None:
        """
        Log estruturado para estatísticas de detecção.
        
        Args:
            logger: Logger a ser usado
            roi_count: Número de ROIs detectadas
            items_count: Número de itens detectados
            divisors_count: Número de divisores detectados
        """
        logger.debug(f"DETECÇÃO - ROI: {roi_count}, Itens: {items_count}, Divisores: {divisors_count}")
    
    @classmethod
    def log_state_transition(cls, logger: logging.Logger, from_state: str, 
                           to_state: str, reason: str = "") -> None:
        """
        Log estruturado para transições de estado.
        
        Args:
            logger: Logger a ser usado
            from_state: Estado anterior
            to_state: Novo estado
            reason: Motivo da transição (opcional)
        """
        reason_text = f" - {reason}" if reason else ""
        logger.info(f"TRANSIÇÃO DE ESTADO: {from_state} → {to_state}{reason_text}")
    
    @classmethod
    def log_layer_completion(cls, logger: logging.Logger, layer: int, 
                           item_count: int, expected_count: int) -> None:
        """
        Log estruturado para conclusão de camadas.
        
        Args:
            logger: Logger a ser usado
            layer: Número da camada
            item_count: Itens contados
            expected_count: Itens esperados
        """
        status = "COMPLETA" if item_count >= expected_count else "INCOMPLETA"
        logger.info(f"CAMADA {layer} {status}: {item_count}/{expected_count} itens")
    
    @classmethod
    def log_error_with_context(cls, logger: logging.Logger, error: Exception, 
                             context: str = "") -> None:
        """
        Log estruturado para erros com contexto.
        
        Args:
            logger: Logger a ser usado
            error: Exceção capturada
            context: Contexto adicional do erro
        """
        context_text = f" - Contexto: {context}" if context else ""
        logger.error(f"ERRO: {type(error).__name__}: {str(error)}{context_text}")
    
    @classmethod
    def log_performance_metrics(cls, logger: logging.Logger, fps: float, 
                              processing_time: float) -> None:
        """
        Log estruturado para métricas de performance.
        
        Args:
            logger: Logger a ser usado
            fps: Frames por segundo
            processing_time: Tempo de processamento em ms
        """
        logger.debug(f"PERFORMANCE - FPS: {fps:.1f}, Tempo: {processing_time:.1f}ms")

# Função de conveniência para inicialização rápida
def init_siac_logging(log_level: str = "INFO", enable_file_logging: bool = True) -> None:
    """
    Função de conveniência para inicializar o sistema de logging.
    
    Args:
        log_level: Nível de log (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        enable_file_logging: Se True, salva logs em arquivo além do console
    """
    SiacLogger.setup_logging(log_level, enable_file_logging)

# Função de conveniência para obter logger
def get_siac_logger(name: str) -> logging.Logger:
    """
    Função de conveniência para obter um logger.
    
    Args:
        name: Nome do logger
        
    Returns:
        Logger configurado
    """
    return SiacLogger.get_logger(name)
