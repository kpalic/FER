package hr.fer.progi.simplicity.entities;

import org.hibernate.annotations.OnDelete;
import org.hibernate.annotations.OnDeleteAction;

import javax.persistence.*;

@Entity
public class Owner extends User{
    @OneToOne(cascade = CascadeType.REMOVE)
    @OnDelete(action = OnDeleteAction.CASCADE)
    @JoinColumn(name = "userCard", referencedColumnName = "id")
    private Card card;


    @OneToOne(cascade = CascadeType.REMOVE)
    @OnDelete(action = OnDeleteAction.CASCADE)
    @JoinColumn(name = "userBusiness", referencedColumnName = "id")
    private Business business;

    public Owner(String username, String email, String password, RoleType role, Card card, Business business) {
        super(username, email, password, role);
        this.card = card;
        this.business = business;
    }

    public Owner() {
    }

    public Card getCard() {
        return card;
    }

    public void setCard(Card card) {
        this.card = card;
    }

    public Business getBusiness() {
        return business;
    }

    public void setBusiness(Business business) {
        this.business = business;
    }
}
