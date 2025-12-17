# Fire Detection YOLOv8

## Информация

- Этот репозиторий был создан на основе проекта [Fire_Detection_YOLOv8_Model_Inference_with_Gradio](https://huggingface.co/spaces/SoulPerforms/Fire_Detection_YOLOv8_Model_Inference_with_Gradio)

- Репозиторий проекта: [https://github.com/whitxowl/fire_detection](https://github.com/whitxowl/fire_detection)

## Описание

Система детекции огня (пожара) с использованием модели YOLOv8. Проект включает:
- Обученную модель YOLOv8 для обнаружения огня на изображениях
- Автоматизированные тесты для проверки корректности детекции
- Docker-контейнер для воспроизводимого запуска

Система может использоваться для раннего обнаружения пожаров, мониторинга территорий, систем безопасности и автоматического оповещения при обнаружении огня.

## Запуск проекта

Этот проект настроен для выполнения в Docker среде.

1. **Клонировать репозиторий**

    ```bash
    git clone https://github.com/whitxowl/fire_detection.git
    cd fire_detection
    ```

2. **Собрать Docker образ**

    ```bash
    docker build -t fire-detection .
    ```

    - После этого запустится процесс сборки, который займет 2-3 минуты

3. **Запустить тесты в контейнере**

    ```bash
    docker run --rm fire-detection
    ```
