import streamlit as st

from app.src.utils import Page


class ViewRecordings(Page):
    def write(self) -> None:
        st.title("View Recordings")

    def render_recordings_on_glasses(self) -> None:
        colms = st.columns(5)
        fields = ["UUID", "Visible Name", "Folder Name", "Created", "Duration"]
        for col, field_name in zip(colms, fields, strict=False):
            # header
            col.write(field_name)

        # for x, email in enumerate(user_table['email']):
        #     col1, col2, col3, col4, col5 = st.columns((1, 2, 2, 1, 1))
        #     col1.write(x)  # index
        #     col2.write(user_table['email'][x])  # email
        #     col3.write(user_table['uid'][x])  # unique ID
        #     col4.write(user_table['verified'][x])   # email status
        #     disable_status = user_table['disabled'][x]  # flexible type of button
        #     button_type = "Unblock" if disable_status else "Block"
        #     button_phold = col5.empty()  # create a placeholder
        #     do_action = button_phold.button(button_type, key=x)
        #     if do_action:
        #          pass # do some action with row's data
        #          button_phold.empty()  #  remove button
