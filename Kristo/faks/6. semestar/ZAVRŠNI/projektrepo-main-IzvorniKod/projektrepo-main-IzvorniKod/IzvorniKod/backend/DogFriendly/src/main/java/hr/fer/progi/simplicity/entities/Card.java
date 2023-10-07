package hr.fer.progi.simplicity.entities;

import org.springframework.format.annotation.DateTimeFormat;

import javax.persistence.*;
import javax.validation.constraints.NotNull;
import javax.validation.constraints.Size;
import java.util.Date;

@Entity
public class Card
{

    @Id
    @GeneratedValue
    private Long id;

    @Column(nullable = false)
    @NotNull
    private String cardNumber;

    @DateTimeFormat(pattern = "yyyy-MM-dd")
    @NotNull
    private Date endDate;

    @NotNull
    @Size(min=3, max=4)
    private String cvv;

    public Card(){}

    public Card(String cardNumber, Date endDate, String cvv)
    {
        this.cardNumber = cardNumber;
        this.endDate = endDate;
        this.cvv = cvv;
    }

    public Long getId()
    {
        return id;
    }

    public void setId(Long id)
    {
        this.id = id;
    }

    public String getCardNumber()
    {
        return cardNumber;
    }

    public void setCardNumber(String cardNumber)
    {
        this.cardNumber = cardNumber;
    }

    public Date getEndDate()
    {
        return endDate;
    }

    public void setEndDate(Date endDate)
    {
        this.endDate = endDate;
    }

    public String getCvv()
    {
        return cvv;
    }

    public void setCvv(String cvv)
    {
        this.cvv = cvv;
    }
}
