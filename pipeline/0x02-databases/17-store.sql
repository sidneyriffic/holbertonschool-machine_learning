-- Subtract data from fields and save it
DELIMITER //
CREATE TRIGGER transaction__ AFTER INSERT ON orders FOR EACH ROW
BEGIN
UPDATE items SET quantity = quantity - NEW.number WHERE NEW.item_name = name;
END//
