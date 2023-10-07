package hr.fer.progi.simplicity.repositories;

import hr.fer.progi.simplicity.entities.Location;
import org.springframework.data.jpa.repository.JpaRepository;

public interface LocationRepository extends JpaRepository<Location, Long> {
    Location findById(long id);
    Location findByName(String name);
}
