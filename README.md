# car_price_prediction
Всем привет, меня зовут Денис Аников 🙋‍♂️

<div id="socials" align="center">
  <a href="https://t.me/akinovdesin">
    <img src="https://img.shields.io/badge/Telegram-blue?style=for-the-badge&logo=telegram&logoColor=white" alt="Telegram"/>
  </a>
  <a href="https://vk.com/akinovdeniska">
    <img src="https://img.shields.io/badge/vkontakte-blue?style=for-the-badge&logo=vk&logoColor=white" alt="Vkontakte"/>
  </a>
  <a href="mailto:daanikov@gmail.com">
    <img src="https://img.shields.io/badge/Gmail-red?style=for-the-badge&logo=gmail&logoColor=white" alt="VK Badge"/>
  </a>
</div>

### Обо мне
- 👷 Обучаюсь на направлении **Бизнес-информатика (цифровая трасформация бизнеса)**
- 📚 Прошел ДПО **Анализ данных и машинное обучение в среде Python**
- 🏫 Активно участую в студенческой жизни, являюсь со-руководителем структуры Project в **ITCenter** в универе
- 🌍 Говорю на: **Русском, English**
- 📫 Reach me by [telegram](https://t.me/akinovdesin), [vkontakte](https://vk.com/akinovdeniska), [gmail](mailto:daanikov@gmail.com")

Бизнес-целью данного проекта является составление прогноза цен на автомобили на основе предложенных данных (https://www.kaggle.com/datasets/hellbuoy/car-price-prediction). В качестве инструментов используются ЯП Python 3.8, а также модули Pandas, Matplotlib, scikit-learn. 

Из цели вытекают следующие задачи, которые были выполнены при создании проекта:

1. Начальное изучение данных. На данном этапе производится описание данных, их исследование, включающее в себя построенеие описательной статистики, проверку на дубликаты, пропуски и тд. Ислледование сопровождается визуализацией.

2. Подготовка данных. Был проведен корреляционный анализ для выбора параметров, которые в дальнейшем используются для построения модели регресии (об этом ниже). Помимо этого, была произведена проверка на выбросы и очистка датасета от них. Графики рассеяния для выбранных параметров присутсвуют.

3. Моделирование. Решение задачи прогнозирования цен на автомобили с использованием моделей регрессии. В качестве моделей регрессии будут рассмотрены методы: линейная регрессия, метод ближайшего соседа, дерево решений, случайный лес.

Метрики оценки точности и качества построенных моделей:

•	Качество модели определяется с использованием коэффициента детерминации (R2), точность модели определяется на основании средней относительной ошибки (MAPE). 

•	Границы значений метрик: R2 должен быть больше либо равен 0.8, MAPE не более 10%.
