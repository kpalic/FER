package hr.fer.progi.simplicity.repositories;

import hr.fer.progi.simplicity.entities.Business;
import org.springframework.data.jpa.repository.JpaRepository;

import java.util.Optional;

public interface BusinessRepository  extends JpaRepository<Business, Long> {
    Optional<Business> findByBusinessName(String businessName);
    Optional<Business> findByBusinessOIB(String businessOIB);
}
