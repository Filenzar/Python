import random

# Класс Hero - базовый класс для всех героев
class Hero:
    def __init__(self, name, level=1, health=100, power=10):
        self.name = name
        self.level = level
        self.health = health
        self.power = power

    def go_on_scouting(self):
        """Определяет, будет ли разведка успешной."""
        success_chance = random.random()  # случайное число от 0 до 1
        return success_chance < 0.6  # шанс успеха

# Класс Warrior - наследуется от Hero, для воинов
class Warrior(Hero):
    def __init__(self, name, level=1, health=120, power=15):
        super().__init__(name, level, health, power)

    def attack(self, enemy_level):
        """Логика атаки воина."""
        print(f"{self.name} атакует противника уровня {enemy_level}")
        return self.level >= enemy_level

# Класс Mage - наследуется от Hero, для магов
class Mage(Hero):
    def __init__(self, name, level=1, health=80, power=10, mana=50):
        super().__init__(name, level, health, power)
        self.mana = mana

    def cast_spell(self, enemy_level):
        """Логика магической атаки."""
        print(f"{self.name} кастует заклинание противника уровня {enemy_level}")
        if self.mana > 10 and self.level >= enemy_level:
            self.mana -= 10  # Расход маны
            return True
        return False
    def mana_reg(self):
        if self.mana<50:
            self.mana+=5

# Класс Kingdom - управление королевством
class Kingdom:
    def __init__(self,food = 100 ,territory = 50):
        self.food = food  # Продовольствие
        self.territory = territory  # Территория


    def feed_people(self,hero,success):
        """Королевство кормит людей, расходуя продовольствие."""
        if success:
            print("Королевство накормило людей!")
            self.food += 20 * hero.level
            self.territory += 10 * hero.level
        else:
            print("Нет достаточно продовольствия!")
            self.food -= 10 * hero.level
            self.territory -= 5 * hero.level

    def escape(self, hero,enemy_level):
        escape_chance = 0.5 + (hero.level - enemy_level) * 0.1 
        return random.random() < escape_chance
    def resources_event(self, success, hero):
        """Управление ресурсами в зависимости от результата разведки."""
        if success:
            # Успешная разведка
            enemy_level = random.randint(1, 5)
            print(f"Разведки прошла успешно, но на пути {hero.name} появляется противник уровня {enemy_level}.")
            action = input("(1 - Драться, 2 - Убежать): ")

            if action == "1":
                if hero.__class__ == Warrior:
                    succ = hero.attack(enemy_level)
                else:
                    succ = hero.cast_spell(enemy_level)
                if succ:
                    print(f"{hero.name} победил врага и успешно вернулся с разведки!")
                    hero.level+=0.2
                    self.feed_people(hero,True)
                    self.show_status()
                else:
                    print(f"{hero.name} потерпел поражение.")
                    self.feed_people(hero,False)
                    hero.level = 3
                    self.show_status()
            if action == "2": 
                if self.escape(hero,enemy_level):
                    print(f"{hero.name} сбежал с продовольствием.")
                    self.feed_people(hero,True)
                else:
                    print(f"{hero.name} сбежал но без продовольствия.")
                    self.feed_people(hero,False)
                self.show_status()
        else:
            # Неудачная разведка
            
            print(f"{hero.name} провалил разведку.")
            self.feed_people(hero,False)
            self.show_status()

    def endgame_event(self):
        """Проверка на бунт среди жителей."""
        if self.food <= 0 or self.territory <= 0:
            print("Жители устроили бунт! Игра завершена.")
            return True
        return False

    def show_status(self):
        """Показывает текущие ресурсы королевства."""
        print(f"Продовольствие: {int(self.food)}, Территория: {int(self.territory)}")

# Главная игровая функция
def main():
    kingdom = Kingdom()

    # Добавляем героев
    warrior = Warrior("Рыцарь Артур", level=3)
    mage = Mage("Маг Гендальф", level=3)

    kingdom.show_status()
    # Игровой цикл
    while True:
        
        action = input(f"Выберите героя для разведки:\n1 - Рыцарь LvL:{round(warrior.level,1)} \n2 - Маг LvL:{round(mage.level,1)} mana:{mage.mana}\nq - Выход): ")

        if action == "q":
            print("Игра завершена.")
            break

        hero = None
        if action == "1":
            hero = warrior
        elif action == "2":
            hero = mage

        if hero:
            # Произошла ли разведка

            kingdom.resources_event(hero.go_on_scouting(), hero)

            # Проверяем на бунт
            if kingdom.endgame_event():
                break
            if (hero!= mage):
                mage.mana_reg()
        


main()
